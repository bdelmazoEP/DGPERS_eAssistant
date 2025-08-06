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
from typing import Optional, Dict
import botocore
import botocore.exceptions

from ep_nlp.askthedocs_config import askthedocs_config
from ep_nlp.bedrock_helper import BedrockHelper

logger = logging.getLogger(__name__)

class Translator:

    def __init__(self, bedrock_client: BedrockHelper):
        # Load configuration once and cache it
        self.cached_config = askthedocs_config.get_configuration()
        self.bedrock_client = bedrock_client

    def translate(self, query: str, language_code: Optional[str] = None) -> Optional[Dict]:
        """
        Translates the input text to the specified language using a Bedrock model.

        Args:
            query (str): The text to translate.
            language_code (Optional[str]): Target language ISO code (default: "en").

        Returns:
            Optional[str]: The translated text, or None if a recoverable error occurred.

        Raises:
            ValueError: If the Bedrock request parameters are invalid.
        """
        target_language = language_code or "en"

        # Get prompt template ID and prepare model request body
        template_id = self.cached_config["ai"].get("prompt_translate_template_id")
        request_body = self.bedrock_client.build_request_body(template_id, [query, target_language])

        try:
            translated_query = self.bedrock_client.invoke_model(request_body)
            logger.info("Text successfully translated to %s.", target_language)
            logger.debug("Translated text:\n%s", translated_query)
            return translated_query

        except botocore.exceptions.ParamValidationError as error:
            raise ValueError(f"Invalid Bedrock parameters: {error}")

        except botocore.exceptions.ClientError as error:
            logger.error("AWS client error during translation: %s", error)
            return None

    def detect_language(self, text: str) -> Optional[Dict]:
        """
        Detects the language of the text received in input

        Args:
            text: Text for which the language is to be detected.

        Returns:
            Dict: The generated answer from the LLM.
        """
        cached_config = askthedocs_config.get_configuration()

        request_body = self.bedrock_client.build_request_body(cached_config["ai"]["prompt_detect_language_template_id"],
                                                              [text[:200]])
        try:
            answer = self.bedrock_client.invoke_model(body=request_body)
            logger.info(f"Language successfully detected.")
        except botocore.exceptions.ClientError as error:
            logger.error(error)
            return None
        except botocore.exceptions.ParamValidationError as error:
            raise ValueError('The parameters you provided are incorrect: {}'.format(error))

        logger.info("\n{0}\n".format(answer))

        return answer
