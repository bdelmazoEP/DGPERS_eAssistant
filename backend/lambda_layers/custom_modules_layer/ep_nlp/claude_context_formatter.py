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
from typing import Dict, List

from ep_nlp.askthedocs_config import askthedocs_config
from ep_nlp.context_formatter import ContextFormatter
import logging

from ep_nlp.utils import remove_non_printable

logger = logging.getLogger(__name__)

class ClaudeContextFormatter(ContextFormatter):

    """
    Formats document retrieval results from a ContextRetriever into a structured XML representation.

    This class transforms a list of retrieved documents into a well-formed XML string,
    specifically designed to be used as context input for an Anthropic Claude LLM (Large Language Model).

    The XML formatting ensures that document metadata and content can be easily parsed
    and integrated into the model's context window, supporting more contextually
    informed and precise language model responses.

    Each document belonging to documents must adhere to this json datastructure:
    [{\"_index\": \"inner-504098497448-index\", \"_id\": \"1%3A0%3AAVLZcZMB3HlcHH2_wWZJ\", \"_score\": 1.44265, \"_source\": {}}, {\"_index\": \"inner-504098497448-index\", \"_id\": \"1%3A0%3AtWrecZMBR74T1-s4yFyZ\", \"_score\": 1.4061766, \"_source\": {}}, {\"_index\": \"inner-504098497448-index\", \"_id\": \"1%3A0%3AFFLncZMB3HlcHH2_lqt8\", \"_score\": 1.3918027, \"_source\": {}}, {\"_index\": \"inner-504098497448-index\", \"_id\": \"1%3A0%3A-2rmcZMBR74T1-s4jHwC\", \"_score\": 1.357709, \"_source\": {}}, {\"_index\": \"inner-504098497448-index\", \"_id\": \"1%3A0%3A0GrhcZMBR74T1-s4gWps\", \"_score\": 1.3404922, \"_source\": {}}, {\"_index\": \"inner-504098497448-index\", \"_id\": \"1%3A0%3AE1LncZMB3HlcHH2_lqt8\", \"_score\": 1.3368905, \"_source\": {}}, {\"_index\": \"inner-504098497448-index\", \"_id\": \"1%3A0%3Al2tTcpMBR74T1-s4IAPR\", \"_score\": 1.2937986, \"_source\": {<custom fields>}}, {\"_index\": \"inner-504098497448-index\", \"_id\": \"1%3A0%3AAFNlcpMB3HlcHH2_wEYp\", \"_score\": 1.2841564, \"_source\": {}}]
}
    Example:
        xml_context = ClaudeContextFormatter.format_context()
        # Resulting XML can be directly used in LLM prompt
    """

    @classmethod
    def format_context(cls, documents: List[Dict]) -> str:
        """
        Formats the documents excerpts listed in documents in an XML string to be used as context of the prompt
        to be submitted to Anthropic Claude LLM.

        """
        documents = cls.replace_metadata_keys(documents)
        context = "<documents>"
        for count, document in enumerate(documents):
            context += f"<document>"
            context += f"<document_id>{count+1}</document_id>"
            document_excerpt = document["_source"]["source"]
            context += f"<document_excerpt>{document_excerpt}</document_excerpt>"
            context += "</document>"
        context += "</documents>"

        logger.info(context)

        return remove_non_printable(context).replace("\\","/").replace("\"", "'")

    @classmethod
    def apply_aliases(cls, document: dict, aliases: List[dict]):
        """
        Applies custom field aliases to the _source field of an Opensearch document.

        Args:
            document (dict): The Opensearch document to apply the aliases to.
            aliases (List[dict]): A list of alias definitions, each containing a 'name' and an 'alias' field.

        Returns:
            dict: The modified document with the aliases applied.
        """

        original_source = document["_source"].copy()  # Make a copy of the original _source
        source = document["_source"]  # Reference to the _source field

        try:
            # Iterate through the list of aliases
            for alias in aliases:
                value = source[alias["name"]]  # Get the value of the original field
                source[alias["alias"]] = value  # Set the value of the alias field
                del source[alias["name"]]  # Remove the original field

            document["_source"] = source  # Update the document's _source field
            return document  # Return the modified document

        except KeyError as err:
            # Log a warning if an error occurs while applying the aliases
            logger.warning(f"An error occurred applying custom field aliases. {str(err)}")
            document["_source"] = original_source  # Revert to the original _source
            return document  # Return the original document

    @classmethod
    def replace_metadata_keys(cls, documents: List[dict]):
        cached_config = askthedocs_config.get_configuration()
        try:
            if len(cached_config["aoss"]["ndx_custom_field_aliases"]) == 0:
                return documents

            aliases = cached_config["aoss"]["ndx_custom_field_aliases"]
            updated_documents = []
            for document in documents:
                updated_document = cls.apply_aliases(document, aliases)
                updated_documents.append(updated_document)
            return updated_documents
        except KeyError as err:
            # Handle missing parameters and log the error
            logger.warning(f"Missing alias for document field or wrong field name. {str(err)} not set.")
            return documents

        return None
    
    @classmethod
    def add_urls_to_documents(cls, documents: List[dict]):
        cached_config = askthedocs_config.get_configuration()
        try:
            enriched_documents = []
            for document in documents:
                document["_source"]["doc_url"] = document["_source"]["x-amz-bedrock-kb-source-uri"] \
                    .replace(cached_config["aws"]["base_s3_uri"], cached_config["aws"]["base_docs_url"])
                enriched_documents.append(document)
                print(document["_source"]["doc_url"])
            logger.info("doc_url field added to all documents")
            return enriched_documents
        except KeyError as err:
            # Handle missing parameters and log the error
            logger.warning(f"Missing document field. {str(err)} not set.")
            return None