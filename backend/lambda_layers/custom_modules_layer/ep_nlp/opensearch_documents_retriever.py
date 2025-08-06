# Copyright 2024 European Parliament.
#
# Licensed under the European Union Public License (EUPL), Version 1.2 or subsequent
# versions of the EUPL (the "Licence"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
# https://eupl.eu/1.2/en/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import logging
import os
from typing import List, Dict, Optional
import boto3
import botocore
import botocore.exceptions

from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from ep_nlp.askthedocs_config import askthedocs_config
from ep_nlp.bedrock_helper import BedrockHelper
from ep_nlp.documents_retriever import DocumentsRetriever
from ep_nlp.query_analyzer import QueryElements, QueryAnalyzer
from ep_nlp.reranker import ReRanker
from ep_nlp.text_to_vector_processor import Text2VectorProcessor
from ep_nlp.utils import initialize_client

# logger.setLevel(logging.INFO)

aws_region = os.getenv("aws_region", askthedocs_config.get_configuration()["aws"]["region"])

class RetrieveContextParams(BaseModel):
    include_fields: Optional[List[str]] = None

class OpensearchDocumentsRetrieverError(Exception):
    pass

class OpensearchDocumentsRetriever(DocumentsRetriever):
    """
    Class used to interact with Amazon OpenSearch Service.
    """

    @staticmethod
    def initialize_opensearch_client(collection_id: str, awsauth) -> OpenSearch:
        """
        Initialize an OpenSearch client for a specific knowledge base.

        Args:
            collection_id (str): The identifier of the opensearch collection.
            awsauth: AWS4Auth authentication credentials.

        Returns:
            OpenSearch: An OpenSearch client instance.
        """

        # Initialize the OpenSearch Serverless client
        client = initialize_client("opensearchserverless", aws_region)
        # Fetch the collection details using the knowledge base ID
        response = client.batch_get_collection(ids=[collection_id])

        if len(response["collectionDetails"]) < 1:
            logger.warning(response["collectionErrorDetails"][0]["errorMessage"])
            raise Exception("{0} id={1}".format(response["collectionErrorDetails"][0]["errorMessage"], collection_id))

        host = (response['collectionDetails'][0]['collectionEndpoint']).replace("https://", "")
        # Initialize the OpenSearch client with the retrieved endpoint
        client = OpenSearch(
            hosts=[{'host': host, 'port': 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=300
        )
        return client

    def __init__(self, session: boto3.session.Session = None,
                 open_search_ndx: str = None,
                 collection_id: str = None,
                 text2Vec: Text2VectorProcessor = None):

        """
        It initializes the EpOpensearchProcessor, which consists of two components: the EpText2VectorProcessor, responsible
        for converting text into a vector using a suitable LLM model (default: amazon.titan-embed-text-v2:0), and the
        EpOpensearchProcessor, which handles the vector search.

        Args:
            session (boto3.session.Session):
            open_search_ndx (str):
            collection_id (str): It refers to the collection id (final part of the ARN for the
            OpenSearch collection that contains the vector index).
            text2Vec (EpText2VectorProcessor): Retrieves the embeddings representing the question's semantic meaning.
            rerank_model_id (Optional[str]): The ID of the rerank model to refine results.

        """
        cached_config = askthedocs_config.get_configuration()
        # Create a new Boto3 session and retrieve AWS credentials
        if session is None:
            session = boto3.Session()

        if collection_id is None:
            #
            # If the environmant variable is not set, attempt to retrieve its value from the configuration file.
            collection_id = os.getenv("aoss_collection_id", cached_config["aoss"]["collection_id"])

        if open_search_ndx is None:
            open_search_ndx = os.getenv("aoss_open_search_ndx", cached_config["aoss"]["open_search_ndx"])

        if text2Vec is None:
            text2Vec = Text2VectorProcessor(os.getenv("aws_region", cached_config["aws"]["region"]),
                                            cached_config["ai"]["bedrock_model_embed"])

        credentials = session.get_credentials()

        awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, "eu-central-1", "aoss",
                           session_token=credentials.token)

        self.open_search_ndx = open_search_ndx
        self.opensearch_client = OpensearchDocumentsRetriever.initialize_opensearch_client(collection_id, awsauth)
        self.text2Vec = text2Vec
        self.rerank_model_id = cached_config.get("aoss").get("rerank_model_id", None)

        logger.info(f'OpensearchDocumentsRetriever initialized {self.open_search_ndx}, {self.rerank_model_id}')

    def retrieve_context(self,
                         query: str,
                         **kwargs) -> List[Dict]:
        """
        Retrieves relevant context based on a query string performing a pure vector search.

        This method:
        1. Converts the query into vector embeddings.
        2. Performs a vector search to retrieve relevant documents.
        3. Optionally reranks the results using a specified rerank model.

        Args:
            query (str): The input query for retrieving context. Language must be French.
            include_fields (Optional[List[str]]): Additional fields to include in search results.

        Returns:
            List[Dict]: A list of dictionaries containing search results.
        """
        # Verify the passed kwargs
        params = RetrieveContextParams(**kwargs)

        # Compute query embeddings
        query_embeddings = self.text2Vec.get_embeddings(query)

        # Determine the number of results based on whether reranking is used
        num_results = 25 if self.rerank_model_id else 8

        # Perform vector search
        results = self.perform_vector_search(query_embeddings, number_of_results=num_results,
                                             include_fields=params.include_fields)

        # If reranking is enabled, apply reranking and log the operation
        if self.rerank_model_id is not None:
            re_ranker = ReRanker(self.rerank_model_id)
            results = re_ranker.rerank_results(query, results, self.rerank_model_id)
            logger.info("Context retrieval results have been re-ranked.")

        return results

    def get_retriever_type(self) -> str:
        return "opensearch"

    def get_knn_search_body(self, max_num_of_results: int, search_vector: [float]) -> Dict:

        cached_config = askthedocs_config.get_configuration()
        # If the environment variable is not set, attempt to retrieve its value from the configuration file.
        vector_field_name = os.getenv("aoss_vector_field_name", cached_config["aoss"]["vector_field_name"])

        return {
            "size": max_num_of_results,
            "query": {
                "knn": {
                    vector_field_name: {
                        "vector": search_vector,
                        "k": max_num_of_results
                    }
                }
            }
        }

    def perform_vector_search(self, vector: [float], number_of_results: int = 5, include_fields: List[str] = None) -> \
    List[dict]:
        """
        Perform a vector search on an OpenSearch index using a K-Nearest Neighbor (KNN) approach.

        Args:
            vector ([float]): The query vector used for similarity search.
            number_of_results (int, optional): The number of top results to return. Defaults to 5.
            include_fields (List[str], optional): A list of specific fields to include in the search results. If None, all fields are returned.

        Returns:
            List[dict]: A list of search hits, each containing the relevant source fields and metadata.
        """
        # Define the base search body for the KNN vector search

        base_search_body = self.get_knn_search_body(number_of_results, vector)
        # If specific fields are requested, add them to the search body
        if include_fields is not None:
            base_search_body["_source"] = include_fields

        search_body = base_search_body  # Assign the prepared search body to a variable

        try:
            # Perform the search using the OpenSearch client
            results = self.opensearch_client.search(index=self.open_search_ndx, body=search_body)

        except botocore.exceptions.ClientError as error:
            # Handle client errors from the OpenSearch service
            raise error

        except botocore.exceptions.ParamValidationError as error:
            # Handle parameter validation errors
            raise ValueError('The parameters you provided are incorrect: {}'.format(error))

        return results["hits"]["hits"]


    def preprocess_query(self, query: str, bedrock_client: BedrockHelper) -> str:
        """
        Uses the available LLM to extract relevant keywords from the query.

        Args:
            query (str): The input text to be analyzed.
            bedrock_client(type): bedrock_client
        Returns:
            str: A space-separated string of extracted keywords or the original query if keywords are not found.
        """
        # query_analysis: Dict = self.analyze_query(query, bedrock_client, prompt_template="extract_keywords")

        query_analyzer: QueryAnalyzer = QueryAnalyzer(bedrock_client)
        query_elements: Optional[QueryElements] = query_analyzer.analyze_query(query, prompt_template="analyze_query")

        if query_elements:
            return ", ".join(query_elements.get_search_elements())
        else:
            return query
