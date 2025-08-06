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
from typing import Dict, List, Optional

import botocore
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

from ep_nlp.bedrock_helper import BedrockHelper
from ep_nlp.askthedocs_config import askthedocs_config

# Set up logging for the application
from ep_nlp.query_analyzer import QueryAnalyzer


###################
# Determine logging level from environment variable or default to INFO
# if os.environ.get("ep_log_level") is None:
#     logger.setLevel(logging.INFO)
# else:
#     logger.setLevel(os.environ.get("ep_log_level"))
#####################

# Custom exception to handle errors specifically related to the chatbot
class ChatbotError(Exception):
    "Custom exception for errors returned by EpChatbot"

    def __init__(self, message):
        self.message = message


# Main Chatbot class to handle question answering logic
class Chatbot:

    def __init__(self,
                 context_retriever_class: type,
                 context_formatter_class: type,
                 aws_region_name: str,
                 model_id: str = None,
                 document_base_url: str = None):
        """
        Initializes the Chatbot class with context retrieval, formatting, and AWS Bedrock client setup.

        Args:
            context_retriever_class (type): A class for retrieving context (e.g., from OpenSearch, AWS Kendra).
            context_formatter_class (type): A class for formatting retrieved context into usable form.
            aws_region_name (str): The AWS region where the Bedrock model is hosted.
            model_id (str, optional): The identifier for the Bedrock LLM model. Defaults to 'amazon.titan-text-express-v1'.
            document_base_url (str, optional): The base URL for accessing documents. Defaults to an S3 bucket URL.
        """
        # Validate provided parameters to ensure they are not None
        Chatbot.validate_params(aws_region_name, model_id, document_base_url)

        # Set class attributes
        self.model_id = model_id
        self.bedrock_client = BedrockHelper(model_id, aws_region_name)
        self.document_base_url = document_base_url

        # Instantiate context retriever and formatter classes
        self.context_retriever = context_retriever_class()
        self.context_formatter = context_formatter_class()

    def get_version(self) -> str:
        pass

    @classmethod
    def validate_params(cls, aws_region_name, model_id, document_base_url):
        """
        Validates key parameters necessary for the chatbot to function.

        Args:
            aws_region_name (str): AWS region where the Bedrock model is located.
            model_id (str): The identifier for the LLM model.
            document_base_url (str): Base URL for accessing documents.

        Raises:
            ClientError: If any of the parameters are not set.
        """
        if model_id is None:
            logger.error("LLM not defined")
            raise ClientError("LLM not defined")

        if document_base_url is None:
            logger.error("document_base_url not defined")
            raise ClientError("document_base_url not defined")

        if aws_region_name is None:
            logger.error("aws_region_name not defined")
            raise ClientError("aws_region_name not defined")

    def generate_answer_text(self, question: str, documents: List[Dict], instructions: [str]) -> Dict:
        """
        Generates an answer to a given question using a Large Language Model (LLM).

        Args:
            question (str): The question to be answered.
            documents (List[Dict]): A list of context documents retrieved by the context retriever.
            instructions ([str]): Instructions to guide the LLM's response.

        Returns:
            Dict: The generated answer from the LLM.
        """

        cached_config = askthedocs_config.get_configuration()
        context = self.context_formatter.format_context(documents)

        query_analyzer: QueryAnalyzer = QueryAnalyzer(self.bedrock_client)

        # Temporarely use of reformulated query.
        # Next version will provide a better management using intent.
        query_analysis_elements = query_analyzer.get_query_intent(question)

        ################
        logger.warning('WARNING WARNING') #################
        logger.warning(f'WHAT EXACTLY IS INSIDE context: {context}') #################
        logger.warning(f'WHAT EXACTLY IS INSIDE query_analysis_elements: {query_analysis_elements}') #################
        ################
        request_body = self.bedrock_client.build_request_body("answer_question", #################### cached_config["ai"]["prompt_template_id"], 
                                                              [context, query_analysis_elements.reformulated_query],
                                                              instruction_values=instructions)
        logger.debug(request_body)

        try:
            answer = self.bedrock_client.invoke_model(request_body)
        except botocore.exceptions.ParamValidationError as error:
            # Raise a ValueError for clearly invalid parameters
            raise ValueError(f"The parameters provided are incorrect: {error}")
        except botocore.exceptions.ClientError as error:
            # Log and recover gracefully from client-level issues
            logger.error("Client error generating answer: %s", error)
            return {}

        return answer

    def answer_question(self, input_params: Dict) -> tuple[dict, list[dict]]:
        """
        Retrieves the context and generates an answer to a user's question.

        Args:
            input_params (Dict): Parameters provided by the user, including the question and instructions.

        Returns:
            Dict: The final generated answer.
        """
        logger.debug("input_params: {0}".format(input_params))
        cached_config = askthedocs_config.get_configuration()

        # Extract question and instructions from input parameters
        question = input_params["question"]
        instructions = input_params["instructions"]

        # Check if the question doesn't violates usage policies.
        query_analyzer: QueryAnalyzer = QueryAnalyzer(self.bedrock_client)

        ################
        # logger.warning('TRACK 1')
        ################
        if query_analyzer.is_query_allowed(question):
            ################
            # logger.warning('TRACK 2')
            ################
            # Retrieve context documents relevant to the question
            retrieved_documents: List[Dict] = self.context_retriever.retrieve_context(question,
                                                                                  include_fields=cached_config["aoss"][
                                                                                      "ndx_custom_fields"])
            logger.info(f"Context successfully retrieved.")

            # Generate the final answer text using the LLM. Return answer and retrieved documents.
            # It is more effective to place the most relevant chunks toward the end of the context when using a RAG
            # (Retrieval-Augmented Generation) approach, especially with decoder-only models like GPT, Claude, or LLaMA.

            return self.generate_answer_text(question, retrieved_documents[::-1], [instructions]), retrieved_documents[::-1]
        else:
            return self.do_not_nswer(question), []

    def do_not_nswer(self, question) -> dict:
        return {"content":[{"text": "Cannot answer to this question."}]}


