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
from typing import List, Optional, Dict, Tuple

import boto3
from botocore.config import Config
import ep_nlp.utils as utils

aws_bedrock_region_name = os.environ.get("AWS_BEDROCK_REGION_NAME")
if aws_bedrock_region_name is None:
    aws_bedrock_region_name = "eu-central-1"

MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

# logging.basicConfig(format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class BedrockHelperError(Exception):
    "Custom exception for errors returned by EpChatbot"

    def __init__(self, message):
        self.message = message


class BedrockHelper:
    """
    https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-text.html
    """

    def __init__(self, model_id: str, aws_region_name):

        config = Config(read_timeout=1000)
        self.bedrock_rt_client = boto3.client(service_name='bedrock-runtime',
                                              region_name=aws_region_name,
                                              config=config)

        self.model_id = model_id

    def close(self):
        self.bedrock_rt_client.close()

    def invoke_model_response(self, body, guardrail_id: str = None, guardrail_version: str = None) -> Dict:
        """
          Invokes a model to generate text based on the given input body.

          Parameters:
          body (str): The input text or data to be processed by the model.
          guardrail_id (str): The unique identifier of the guardrail used to evaluate the query.
          guardrail_version (str): The version of the guardrail used to evaluate the query.
          Returns:
          dict: The response returned by model invocation.

          """

        contentType = 'application/json'
        accept = 'application/json'

        kwargs: Dict = {
            "modelId": self.model_id,
            "contentType": contentType,
            "accept": accept,
            "body": body
        }

        if guardrail_id is not None:
            kwargs["guardrailIdentifier"] = guardrail_id
            kwargs["guardrailVersion"] = guardrail_version

        return self.bedrock_rt_client.invoke_model(**kwargs)


    def invoke_model(self, body, guardrail_id: str = None, guardrail_version: str = None) -> Dict:
        """
          Invokes a model to generate text based on the given input body.

          Parameters:
          body (str): The input text or data to be processed by the model.
          guardrail_id (str): The unique identifier of the guardrail used to evaluate the query.
          guardrail_version (str): The version of the guardrail used to evaluate the query.
          Returns:
          dict: The response body containing the generated text or other relevant data.

          Raises:
          EpBedrockError: If there is an error during text generation.
          """
        response: Dict = self.invoke_model_response(body, guardrail_id=guardrail_id, guardrail_version=guardrail_version)
        response_body = json.loads(response.get("body").read())

        finish_reason = response_body.get("error")
        if finish_reason is not None:
            raise BedrockHelperError(f"Text generation error. Error is {finish_reason}")

        logger.info(
            "Successfully generated text with model %s", self.model_id)

        return response_body

    def valid_messages(self, additional_messages, model_id) -> bool:

        return True

    def extract_token_counts(self, response: Dict) -> Tuple[int, int]:
        return int(response["usage"]["inputTokens"]), int(response["usage"]["outputTokens"])

    def build_request_body(self,
                           request_body_template_id: str,
                           values: List[str],
                           instruction_values=None) -> Optional[str]:
        """
        Constructs the request body for invoking a language model (LLM) based on the specified prompt template and model.

        Args:
            request_body_template_id (str): The identifier for the prompt template.
            values (List[str]): The values to instantiate the template with.
            instruction_values (Optional[List[str]]): Optional instruction values for the template.

        Returns:
            Optional[str]: A JSON-formatted string containing the request body for the LLM, or None if the template is not found.
        """

        ############ vvv
        request_body_template = utils.get_request_body_template(self.model_id, request_body_template_id)
        if request_body_template:
            logger.warning(f"WHAT EXACTLY IS INSIDE request_body_template_id: {request_body_template_id}") #############
            logger.warning(f"WHAT EXACTLY IS INSIDE values: {values}") #############
            logger.warning(f"WHAT EXACTLY IS INSIDE request_body_template: {request_body_template}") ################
            prompt = request_body_template.get("PROMPT").format(*values)
            if instruction_values and len(instruction_values) > 0:
                system = request_body_template.get("SYSTEM").format(*instruction_values)
            else:
                system = request_body_template.get("SYSTEM")

            # request_body = request_body_template["REQUEST_BODY"].replace("$system$", system).replace("$prompt$", prompt) ############ no longer works because JSON
            ###### different JSON handling vv
            request_body = request_body_template['REQUEST_BODY']
            if 'system' in request_body.keys():
                request_body['system'] = system
            request_body['messages'][0]['content'] = prompt
            ###### ^^
            logger.debug(request_body)

            return json.dumps(request_body).encode('utf-8') ################ added this otherwise as a dict it gives problems
        else:
            logger.error("No request body info were found. Check the configuration file.")
            ############ ^^^

        return None
