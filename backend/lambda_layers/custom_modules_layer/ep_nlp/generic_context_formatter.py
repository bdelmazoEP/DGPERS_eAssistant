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

import os
from typing import Dict, List
from ep_nlp.context_formatter import ContextFormatter
from ep_nlp.askthedocs_config import askthedocs_config
import logging

from ep_nlp.utils import remove_non_printable

logger = logging.getLogger(__name__)

class GenericContextFormatter(ContextFormatter):

    @classmethod
    def format_context(cls, documents: List[Dict]) -> str:
        """
        Formats the documents excerpts listed in documents in an XML string to be used as context of the prompt
        to be submitted to Anthropic Claude LLM.

        If custom fields are specified, for aeach field a specific xml tag is added to the xml definition of the
        document.

        """
        cached_config = askthedocs_config.get_configuration()
        # If the environmant variable is not set, attempt to retrieve its value from the configuration file.
        custom_fields = os.getenv("aoss_ndx_custom_fields", cached_config["aoss"]["ndx_custom_fields"])

        context = "<documents>"
        for count, document in enumerate(documents):
            context += f"<document>"
            context += f"<document_id>{count+1}</document_id>"
            for field_key in custom_fields:
                field_value = document["_source"].get(field_key, "unknown")
                if field_value != "unknown":
                    context += f"<{field_key}>{field_value}</{field_key}>"
                else:
                    logger.warning(f"custom field {field_key} not found.")

            document_excerpt = document["_source"]["AMAZON_BEDROCK_TEXT_CHUNK"]\
                                        .replace("\"", "'")\
                                        .replace("\r", "")\
                                        .replace("\f", "")
            context += f"<document_excerpt>{document_excerpt}</document_excerpt>"
            context += "</document>"
        context += "</documents>"

        logger.info(context)

        return remove_non_printable(context).replace("\\","/")