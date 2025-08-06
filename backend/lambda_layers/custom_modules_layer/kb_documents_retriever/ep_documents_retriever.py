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
from typing import List, Dict, Optional

import botocore
import boto3
from botocore.exceptions import ParamValidationError
from ep_nlp.translator import Translator
from pydantic import BaseModel

from ep_nlp.opensearch_documents_retriever import OpensearchDocumentsRetriever, Text2VectorProcessor
from ep_nlp.askthedocs_config import askthedocs_config
from ep_nlp.reranker import ReRanker
from ep_nlp.utils import import_class

logger = logging.getLogger(__name__)


class EpDocumentsRetrieveContextParams(BaseModel):
    include_fields: Optional[List[str]] = None


class EpDocumentsRetriever(OpensearchDocumentsRetriever):
    """
    An implementation of the OpensearchDocumentsRetriever specialized for European Parliament documents.

    This retriever supports hybrid retrieval using both BM25 and vector-based similarity search.
    Optionally, it can rerank the results using a Bedrock-compatible reranker model.

    Args:
        session (boto3.session.Session, optional): An optional boto3 session for AWS service clients.
        open_search_ndx (str, optional): The name of the OpenSearch index to query.
        collection_id (str, optional): ID of the document collection used in the retriever.
        text2Vec (Text2VectorProcessor, optional): Component used to convert text queries into vector embeddings.
    """

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
        params = EpDocumentsRetrieveContextParams(**kwargs)

        # Step 1: Embed the query for vector search
        query_embeddings = self.text2Vec.get_embeddings(query)

        # Step 2: Decide number of results to fetch
        num_results = 20 if self.rerank_model_id else 8

        # Step 3: Perform vector (KNN) search
        results_knn = self.perform_vector_search(
            query_embeddings,
            number_of_results=num_results,
            include_fields=params.include_fields
        )

        # Step 4: Translate the query and extract entities
        translated_query = query
        try:
            translation_result = self.translate_to_french(query)
            translated_query = translation_result.get("content", [{}])[0].get("text", query)
        except Exception as e:
            logger.warning(f"Query translation failed, using original query. Error: {e}")

        # Step 5: Preprocess translated query to extract entities
        query_entities = self.preprocess_query(translated_query, self.bedrock_client)
        logger.info(f"Found named entities: {query_entities}")

        # Step 6: Perform BM25 search with extracted entities
        results_bm25 = self.perform_bm25_search(
            query_entities,
            number_of_results=5,
            include_fields=params.include_fields
        )

        # Step 7: Deduplicate results (merge KNN and BM25)
        combined_results = results_knn + results_bm25
        distinct_results = EpDocumentsRetriever.__get_distinct_values(combined_results)

        print(json.dumps(distinct_results))
        # Step 8: Rerank if model is configured
        if self.rerank_model_id:
            logger.info(f"Re-ranking with model: {self.rerank_model_id}")
            re_ranker = ReRanker(self.rerank_model_id)
            distinct_results = re_ranker.rerank_results(translated_query, distinct_results, self.rerank_model_id)
            logger.info("Results have been re-ranked.")

        return distinct_results

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

    def translate_to_french(self, query: str) -> Optional[str]:
        """
        Translates the input text to French using a Bedrock model.

        Args:
            language_code(str): Destination language ISO code.
            query (str): The text to translate.

        Returns:
            str: The translated text, or None if a recoverable error occurred.

        Raises:
            ValueError: If the parameters provided to the Bedrock model are invalid.
        """
        return self.translator.translate(query, language_code="fr")

    def get_retriever_type(self) -> str:
        """
        Returns the type of retriever.

        Returns:
            str: The retriever type, e.g., "ep".
        """
        return "ep"

    @classmethod
    def isPresent(cls) -> bool:
        """
        Indicates whether this retriever is present/enabled.

        Returns:
            bool: Always returns True.
        """
        return True

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
