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
import json
import logging
import os
from typing import List, Dict, Optional

import boto3
import botocore

from botocore.client import BaseClient
import pprint as pp

from ep_nlp.translator import Translator
from pydantic import BaseModel, Field

from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

from ep_nlp.text_to_vector_processor import Text2VectorProcessor
from ep_nlp.opensearch_documents_retriever import OpensearchDocumentsRetriever
from ep_nlp.askthedocs_config import askthedocs_config
from ep_nlp.reranker import ReRanker
from ep_nlp.utils import import_class


def initialize_client(client_type: str) -> BaseClient:
    """
    Initialize an AWS service client using boto3.

    Args:
        client_type (str): The type of AWS service client to initialize (e.g., 'bedrock-runtime').

    Returns:
        BaseClient: A boto3 client for the specified AWS service.

    Raises:
        ClientError: If there is an error with the AWS service client.
        ValueError: If the provided parameters are incorrect.
    """

    try:
        cached_config = askthedocs_config.get_configuration()

        # If the environmant variable is not set, attempt to retrieve its value from the configuration file.
        aws_region = os.getenv("aws_region", cached_config["aws"]["region"])
        client = boto3.client(client_type, region_name=aws_region)
    except botocore.exceptions.ClientError as error:
        # Handle AWS client errors
        raise error
    except botocore.exceptions.ParamValidationError as error:
        # Handle parameter validation errors
        raise ValueError('The parameters you provided are incorrect: {}'.format(error))

    return client

def convert_docs(docs: List[Document]) -> List[dict]:

    converted_docs: List[dict] = []
    for doc in docs:
        metadata = doc.metadata
        chunk_text = doc.page_content
        source = metadata["source_metadata"]
        source["AMAZON_BEDROCK_TEXT_CHUNK"] = chunk_text
        converted_doc = {
            "_index": "not available",
            "_id": source["x-amz-bedrock-kb-chunk-id"],
            "_score": metadata["score"],
            "_source": source,
        }
        converted_docs.append(converted_doc)

    return converted_docs


class KnowledgebaseRetrieveContextParams(BaseModel):
    include_fields: Optional[List[str]] = None


class KnowledgebaseDocumentsRetriever(OpensearchDocumentsRetriever):

    def __init__(self, session: boto3.session.Session = None,
                 open_search_ndx: str = None,
                 collection_id: str = None,
                 text2Vec: Text2VectorProcessor = None):

        super().__init__(session, open_search_ndx, collection_id, text2Vec)
        self.cached_config = askthedocs_config.get_configuration()

        bedrock_helper_class = import_class(self.cached_config["ai"]["bedrock_helper_module"],
                                            self.cached_config["ai"]["bedrock_helper_class"])
        self.bedrock_client = bedrock_helper_class(self.cached_config["ai"]["bedrock_model_generate"],
                                                   self.cached_config["aws"]["region"])
        self.rerank_model_id = self. cached_config.get("aoss").get("rerank_model_id", None)
        self.knowledge_base_id = os.getenv("local_knowledgebase_id", self.cached_config["local"]["knowledgebase_id"])
        self.overrideSearchType = "SEMANTIC"
        self.kkn_max_num_of_results = os.getenv("local_knowledgebase_number_of_results", self.cached_config["local"]["knowledgebase_number_of_results"])
        self.bedrock_agent_runtime_client=initialize_client("bedrock-agent-runtime")
        self.translator = Translator(self.bedrock_client)

    def retrieve_context(
            self,
            query: str,
            **kwargs
    ) -> List[Dict]:
        """
        Retrieves relevant documents from OpenSearch using both vector and BM25 search.
        Optionally reranks and deduplicates the results.

        Args:
            query (str): The user query string.
            include_fields (List[str], optional): Specific fields to include in the search response.

        Returns:
            List[Dict]: A list of relevant documents.
        """
        # Verify the passed kwargs
        params = KnowledgebaseRetrieveContextParams(**kwargs)

        kkn_max_num_of_results = round(self.kkn_max_num_of_results*2.5) if self.rerank_model_id else self.kkn_max_num_of_results
        bm25_max_num_of_results = self.kkn_max_num_of_results
        logger.debug(f"kkn_max_num_of_results={kkn_max_num_of_results}, bm25_max_num_of_results{bm25_max_num_of_results}")

        translated_query = query
        results_knn = []

        ################
        # logger.warning('TRACK 5')
        ################
        try:
            translation_result = self.translate_to_french(query)
            translated_query = translation_result.get("content", [{}])[0].get("text", query)
            results_knn = self.perform_knn_search(
                translated_query,
                max_num_of_results=kkn_max_num_of_results,
                include_fields=params.include_fields
            )
        except Exception as e:
            logger.warning(f"Query translation failed, using original query. Error: {e}")

        ################
        # logger.warning('TRACK 6')
        ################
        query_entities = self.preprocess_query(translated_query, self.bedrock_client)
        logger.info(f"Found named entities: {query_entities}")

        results_bm25 = self.perform_bm25_search(
            query_entities,
            number_of_results=bm25_max_num_of_results,
            include_fields=params.include_fields
        )

        combined_results = results_knn + results_bm25
        distinct_results = KnowledgebaseDocumentsRetriever.__get_distinct_values(combined_results)

        ################
        # logger.warning('TRACK 7')
        ################

        # Rerank if model is configured
        if self.rerank_model_id:
            logger.info(f"Re-ranking with model: {self.rerank_model_id}")
            re_ranker = ReRanker(self.rerank_model_id)
            distinct_results = re_ranker.rerank_results(translated_query, distinct_results, self.rerank_model_id)
            logger.info("Results have been re-ranked.")

        ################
        # logger.warning('TRACK 8')
        ################

        return distinct_results


    def perform_bm25_search(self, query: str, number_of_results: int = 10,
                            include_fields: List[str] = None) -> List[dict]:
        """
        Executes a BM25 keyword search on OpenSearch.

        Args:
            query (str): The query string to search for.
            number_of_results (int): Number of top documents to retrieve.
            include_fields (List[str], optional): Fields to include in the response.

        Returns:
            List[dict]: A list of document hits.
        """
        search_body = self.get_bm25_search_body(number_of_results, query)

        if include_fields is not None:
            search_body["_source"] = include_fields

        try:
            results = self.opensearch_client.search(index=self.open_search_ndx, body=search_body)
        except botocore.exceptions.ClientError as error:
            raise error
        except botocore.exceptions.ParamValidationError as error:
            raise ValueError(f'perform_bm25_search: The parameters you provided are incorrect: {error}')

        return results["hits"]["hits"]

    def get_bm25_search_body(self, max_num_of_results: int, query: str) -> Dict:
        """
        Constructs the OpenSearch query body for BM25 search.

        Args:
            max_num_of_results (int): Maximum number of results to retrieve.
            query (str): Query string for which results are retrieved.

        Returns:
            Dict: OpenSearch query dictionary for BM25 search.
        """
        # return {
        #     "query": {
        #         "multi_match": {
        #             "query": query,
        #             "fields": [
        #                 "AMAZON_BEDROCK_TEXT_CHUNK",
        #                 "titulo",
        #                 "longtitle",
        #                 "subtype"
        #             ],
        #             "type": "best_fields",
        #             "operator": "or"
        #         }
        #     }
        # }
        return {
                "query": {
                "multi_match": {
                  "query": query,
                  "fields": ["AMAZON_BEDROCK_TEXT_CHUNK", "longtitle"],
                  "type": "best_fields",
                  "operator": "or",
                  "fuzziness": "AUTO"
                }
              },
              "size": max_num_of_results,
              "from": 0
            }

    def perform_knn_search(self, query, max_num_of_results=8,include_fields=None) -> List[Dict]:
        # See https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_KnowledgeBaseVectorSearchConfiguration.html
        retriever = AmazonKnowledgeBasesRetriever(
            client=self.bedrock_agent_runtime_client,
            knowledge_base_id=self.knowledge_base_id,
            retrieval_config={"vectorSearchConfiguration":
                                  {"numberOfResults": max_num_of_results,
                                   'overrideSearchType': self.overrideSearchType,
                                   }
                              }
                            )

        docs: [Document] = retriever.invoke(
            input=query
        )

        return convert_docs(docs)

    def translate_to_french(self, query: str) -> Optional[str]:
        """
        Translates the input text to French using a Bedrock model.

        Args:
            query (str): The text to translate.

        Returns:
            str: The translated text, or None if a recoverable error occurred.

        Raises:
            ValueError: If the parameters provided to the Bedrock model are invalid.
        """
        return self.translator.translate(query, language_code="fr")

    def get_retriever_type(self) -> str:
        return "knowledgebase"

    @classmethod
    def __get_distinct_values(cls, lst: List[Dict]) -> List[Dict]:
        """
        Removes duplicate documents based on their '_id' field.

        Args:
            lst (List[Dict]): List of document hits possibly containing duplicates.

        Returns:
            List[Dict]: Deduplicated list of documents.
        """
        seen = set()
        unique = []
        for d in lst:
            _id = d.get('_id')
            if _id not in seen:
                seen.add(_id)
                unique.append(d)
        return unique

    @classmethod
    def isPresent(cls) -> bool:
        return True

def main():
    retriever = KnowledgebaseDocumentsRetriever(knowledge_base_id="TDOLXGSAIQ",
                                                overrideSearchType="HYBRID",
                                                numberOfResults=25)

    docs = retriever.retrieve_context("Quand le traité de Maastricht a-t-il été signé ?")

    for doc in docs:
        pp.pprint(doc)

if __name__ == "__main__":
    main()