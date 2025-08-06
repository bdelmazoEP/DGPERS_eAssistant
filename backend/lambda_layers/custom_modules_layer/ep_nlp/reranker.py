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
import json
from typing import List, Dict

from ep_nlp.askthedocs_config import askthedocs_config
from ep_nlp.utils import initialize_client


logger = logging.getLogger(__name__)

class ReRanker():

    def __init__(self, model_id:str):

        self.rerank_model_id: str = model_id

    def rerank_results(self, query: str, results: List[Dict], model_id: str) -> List[Dict]:
        """
        Reranks the search results using a Bedrock model.

        :param query: The search query.
        :param results: A list of search results containing "_source" data.
        :param model_id: The Bedrock model ID for reranking.
        :return: A reranked list of search results.
        """
        if not results:
            logger.warning("No results provided for reranking.")
            return []

        # Extract documents from results
        documents = [hit["_source"]["AMAZON_BEDROCK_TEXT_CHUNK"] for hit in results]
        logger.info("Extracted %d documents for reranking.", len(documents))

        payload = json.dumps(
            {
                "query": query,
                "documents": documents,
                "top_n": 8,
                "api_version": 2,
            }
        )

        aws_region = os.getenv("aws_region", askthedocs_config.get_configuration()["aws"]["region"])
        bedrock_client = initialize_client("bedrock-runtime", aws_region)

        # Invoke the Bedrock model
        response =bedrock_client.invoke_model(
            modelId=self.rerank_model_id,
            accept="application/json",
            contentType="application/json",
            body=payload,
        )

        response_body = json.loads(response.get("body").read())
        logger.info("Received reranking response: %s", json.dumps(response_body, indent=2))

        top_score = response_body["results"][0].get("relevance_score", 0)
        threshold = top_score * 0.40

        # Extract the indices of the matching documents
        matching_indices = [result["index"] for result in response_body["results"] if result["relevance_score"] > threshold]

        # Retrieve and print the matching documents
        reranked_results = [results[i] for i in matching_indices]
        logger.info("Final reranked results count: %d", len(reranked_results))

        return reranked_results
