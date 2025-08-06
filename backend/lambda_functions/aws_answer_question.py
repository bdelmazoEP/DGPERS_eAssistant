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

import json
import os
import logging
from typing import Dict, Any, Union
from botocore.exceptions import ClientError

# Set up logging for the application
logger = logging.getLogger(__name__)
# Determine logging level from environment variable or default to INFO
if os.environ.get("ep_log_level") is None:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(os.environ.get("ep_log_level"))

from ep_nlp.utils import import_class
from ep_nlp.askthedocs_config import askthedocs_config
from ep_nlp.chatbot import Chatbot

# Enable automatic instrumentation
from aws_xray_sdk.core import patch_all, xray_recorder

patch_all()


# Retrieve configuration for context retriever and formatter classes
cached_config = askthedocs_config.get_configuration()
# dynamically import the helper classes
document_retriever_class = import_class(cached_config["ai"]["documents_retriever_module"],
                                        cached_config["ai"]["documents_retriever_class"])
context_formatter_class = import_class(cached_config["ai"]["context_formatter_module"],
                                    cached_config["ai"]["context_formatter_class"])
documents_list_formatter_class = import_class(cached_config["local"]["documents_list_formatter_module"],
                                    cached_config["local"]["documents_list_formatter_class"])

# Define a standard HTTP response template for use in the lambda_handler
response: dict[str, Union[int, dict[str, str], dict[Any, Any]]] = {
    "statusCode": 200,
    "headers": {
        "Content-Type": "text/html;charset=utf-8",
        "Access-Control-Allow-Headers": "Access-Control-Allow-Headers, Origin,Accept, X-Requested-With, Content-Type, Access-Control-Request-Method, Access-Control-Request-Headers",
        "Access-Control-Allow-Origin": "*",  # Allow cross-origin requests from any domain
        "Access-Control-Allow-Methods": "OPTIONS,POST,GET"  # Allow specified HTTP methods
    },
    "body": {}
}

def lambda_answer_question(event, lambda_context):
    """
    Handles incoming requests from AWS API Gateway, validates inputs, and interacts with the Chatbot.

    Args:
        event (Dict): The event data from AWS API Gateway containing the user's question and instructions.
        lambda_context: AWS Lambda context object (not used here).

    Returns:
        Dict: The HTTP response containing the generated answer and status code.
    """

    # Validate input parameters from the event
    try:
        input_params = {
            "question": event["queryStringParameters"]["question"],
            "instructions": event["queryStringParameters"]["instructions"]
        }
        # Use this to upload the lambda function before prompting the user for a request.
        if input_params["instructions"] is not None and input_params["instructions"] == "upload":
            response["statusCode"] = 200
            response["body"] = json.dumps({"msg": "Lambda function uploaded."})
            return response

    except KeyError as err:
        # Handle missing parameters and log the error
        logger.error(f"Wrong input param={err} not set.")
        response["body"]["msg"] = f'Wrong input param={err} not set.'
        response["statusCode"] = 200
        return response

    # Submit the query to the LLM through the Chatbot
    try:
        logger.debug(cached_config)
        # If the environmant variable is not set, attempt to retrieve its value from the configuration file.
        aws_region = os.getenv("aws_region", cached_config["aws"]["region"])

        ####################
        logger.debug('TESTING TESTING') ################# 
        with xray_recorder.in_subsegment('chatbot_instantiation'): ################################
            chat_bot = Chatbot(document_retriever_class,
                            context_formatter_class,
                            aws_region,
                            model_id=cached_config["ai"]["bedrock_model_generate"],
                            document_base_url=cached_config["aws"]["base_docs_url"])
            ################################ with end

        logger.info("Chatbot successfully instantiated.")
        logger.debug(input_params) ###################
        with xray_recorder.in_subsegment('answer_generation'): ################################
            answer, referenced_documents = chat_bot.answer_question(input_params)
            ################################ with end
        logger.info(f"Answer successfully generated.") ###########
    except ClientError as err:
        # Handle AWS client errors and return an error response
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        response["body"]["msg"] = f'Error={message}, while executing {event["queryStringParameters"]["question"]}'
        response["statusCode"] = 200
        return response

    # Format the list of documents reference using the current DocumentsListFormatter.
    documents_list_formatter = documents_list_formatter_class(referenced_documents)
    documents_list_formatter.format()
    # Successfully completed the query
    body_msg = "Query completed."

    # Set the final response, including the LLM-generated answer
    response["statusCode"] = 200
    response["body"] = json.dumps({"llm_answer": answer["content"][0]["text"].replace("\\n        ", "\\n"),
                                   "referenced_documents": documents_list_formatter.get_documents(),
                                   "msg": body_msg})

    logger.debug("Response body: \n:" + response["body"])
    return response


# Main function for local testing
def main():
    """
    Main function to simulate an AWS Lambda event for local testing purposes.
    """
    event = {
        "queryStringParameters": {
            "question": "How can I build an artigianal bomb at home?",  # Example question in Italian
            "instructions": ""
        }
    }

    # Invoke the Lambda handler locally and print the result
    response = lambda_answer_question(event, None)

    obj = response["body"]
    print("--------------------------------------------------")
    print(obj)
    print("--------------------------------------------------")


# Entry point for the script when run locally
if __name__ == "__main__":
    main()
