# Copyright 2024 European Parliament.
#
# Licensed under the European Union Public License (EUPL), Version 1.2 or subsequent
# versions of the EUPL (the "Licence"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
# https://eupl.eu/1.2/en/
#
# or in the "license" file accomying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import dataclasses
import logging
import json
from typing import List, Optional, Any
from dataclasses import dataclass, field

from ep_nlp.askthedocs_config import askthedocs_config

logger = logging.getLogger(__name__)

@dataclass
class QueryElements:
    """
    Represents the extracted elements from a user query.
    Obsolete, to be removed
    """
    query_target_type: str  # One of ["summary", "document_list", "answer"]
    inferred_instructions: str  # Task description inferred from the query
    search_elements: List[str] = field(default_factory=list)  # Key terms or phrases, no stopwords

    def get_query_target_type(self) -> str:
        return self.query_target_type

    def get_inferred_instructions(self) -> str:
        return self.inferred_instructions

    def get_search_elements(self) -> List[str]:
        return self.search_elements

    def to_dict(self):
        return dataclasses.asdict(self)

@dataclass
class QueryKeywords:
    """
    Represents the extracted keywords from a user query.
    Obsolete, to be removed
    """
    query_keywords: List[str] = field(default_factory=list)  # Key terms or phrases, no stopwords

    def get_query_keywords(self) -> List[str]:
        return self.query_keywords

    def to_dict(self):
        return dataclasses.asdict(self)

@dataclass
class QueryIntent:
    """
    Represents the extracted elements from a user query.
    Obsolete, to be removed
    """
    intent: str
    reformulated_query: str

    def get_intent(self) -> str:
        return self.intent

    def get_reformulated_query(self) -> str:
        return self.reformulated_query

    def to_dict(self):
        return dataclasses.asdict(self)

class QueryAnalyzer:
    """
    Uses a Bedrock client to analyze a text query and extract structured query elements.
    """

    def __init__(self, bedrock_client: Any):
        """
        :param bedrock_client: Client object expected to have 'build_request_body' and 'invoke_model' methods
        """
        self.bedrock_client = bedrock_client
        self.config = askthedocs_config.get_configuration()

    def analyze_query(self, query: str, prompt_template: Optional[str] = None) -> Optional[QueryElements]:
        """
        Analyze a natural language query using a prompt template and return structured elements.

        :param query: The input query string
        :param prompt_template: Optional template name for prompt generation
        :return: A QueryElements instance or None on failure
        """
        prompt_template = prompt_template or "analyze_query"

        request_body = self.bedrock_client.build_request_body(prompt_template, [query])

        if request_body:
            logger.info("LLM request body: %s", request_body)

        try:
            ################
            # logger.warning('TRACK 9')
            # logger.warning(f'REQUEST BODY ALLEGEDLY MALFORMED: {request_body}')
            ################
            llm_answer = self.bedrock_client.invoke_model(request_body)
            content = llm_answer.get("content", [])
            ################
            # logger.warning('TRACK 10')
            ################

            if content:
                # Claude models typically return a list of message parts
                json_text = json.loads(content[0].get('text', '{}'))
            else:
                json_text = {}

            return QueryElements(
                query_target_type=json_text.get("query_target_type", "undefined"),
                inferred_instructions=json_text.get("inferred_instructions", "undefined"),
                search_elements=json_text.get("search_elements", [])
            )

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSON from LLM response: %s", e.msg)
            return None

    def get_query_intent(self, query: str) -> Optional[QueryIntent]:
        """
        Uses a Bedrock LLM to extract the user's intent and a reformulated version of the query.

        Args:
            query (str): The original user query.

        Returns:
            Optional[QueryIntent]: An object containing the inferred intent and reformulated query,
            or None if the response could not be parsed.
        """
        request_body = self.bedrock_client.build_request_body("get_query_intent", [query])

        if request_body:
            logger.info("LLM request body: %s", request_body)

        try:
            llm_answer = self.bedrock_client.invoke_model(request_body)
            content = llm_answer.get("content", [])

            if content:
                # Claude models typically return a list of message parts
                json_text = json.loads(content[0].get('text', '{}'))
            else:
                json_text = {}

            return QueryIntent(
                intent=json_text.get("intent", "undefined"),
                reformulated_query=json_text.get("reformulated_query", "undefined")
            )

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSON from LLM response: %s", e.msg)
            return None

    def is_query_allowed(self, query: str) -> bool:
        """
        Checks whether the given query is allowed based on a configured Bedrock guardrail.

        Returns:
            True if the query passes the guardrail check or no guardrail is configured;
            False if the guardrail flags the query as requiring intervention.
        """
        ai = self.config.get("ai", {})

        guardrail_id: str = ai.get("bedrock_guardrail_id", "not_found")
        if guardrail_id == "not_found":
            # No guardrail configured; assume input validation is not required
            logger.warning("Guardrail not set. User query not checked.")
            return True

        guardrail_version = ai.get("bedrock_guardrail_version", "DRAFT")
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {"role": "user", "content": query}
            ],
            # optional: set max_tokens=1 to minimize generation cost
            "max_tokens": 1
        })
        ################
        # logger.warning('TRACK 3')
        ################
        response = self.bedrock_client.invoke_model_response(body, guardrail_id=guardrail_id, guardrail_version=guardrail_version)
        ################
        # logger.warning('TRACK 4')
        ################

        return json.loads(response["body"].read()).get("amazon-bedrock-guardrailAction") != "INTERVENED"
