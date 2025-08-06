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
from typing import List, Dict

from ep_commons.presigned_url_generator import PresignedUrlGenerator, PresignedUrlGenerationError
from ep_nlp.askthedocs_config import askthedocs_config
from ep_nlp.documents_list_formatter import DocumentsListFormatter

logger = logging.getLogger(__name__)

class GenericDocumentsListFormatter(DocumentsListFormatter):

    def __init__(self, documents: List[Dict]):

        self.documents = documents

    def get_documents(self) -> List[Dict]:
        return self.documents

    def set_documents(self, documents: List[Dict]):
        self.documents = documents

    def format(self) -> List[Dict]:
        enriched_documents = self.__add_urls_to_documents(self.documents)
        return self.__add_display_title_to_documents(enriched_documents)

    @classmethod
    def __add_urls_to_documents(cls, documents: List[Dict]):
        """
        Enriches a list of documents by adding presigned URLs to the 'doc_url' field.

        This method:
        - Retrieves configuration settings.
        - Extracts the S3 bucket name from the base S3 URI.
        - Initializes a presigned URL generator.
        - Iterates through the documents to replace the base S3 URI with a base URL.
        - Generates a presigned URL for each document and updates its 'doc_url' field.
        - Logs relevant information and handles errors gracefully.

        Args:
            documents (List[Dict]): A list of document dictionaries, where each document
                                    contains an '_source' key with metadata.

        Returns:
            List[Dict]: The list of enriched documents with updated 'doc_url' fields.
                        If an error occurs, the original documents are returned.
        """

        cached_config = askthedocs_config.get_configuration()

        extract_bucket_name = lambda uri: uri.split("://")[1].split("/")[0]
        extract_object_key = lambda uri: "/".join(uri.split("://")[1].split("/")[1:])

        try:
            # Initialize the presigned URL generator using the extracted S3 bucket name
            presigned_url_generator = PresignedUrlGenerator(
                extract_bucket_name(cached_config["aws"]["base_s3_uri"])
            )

            enriched_documents = []
            for document in documents:

                # Generate a presigned URL with a 1200-second expiration
                object_key = extract_object_key(document["_source"]["x-amz-bedrock-kb-source-uri"])
                presigned_url = presigned_url_generator.generate_presigned_url(
                                                                object_key,
                                                                expiration=1200)


                # Update the document with the generated presigned URL

                document["_source"]["doc_url"] = presigned_url
                enriched_documents.append(document)

                logger.debug("{0} {1} {2}".format(
                    document["_source"]["doc_url"],
                    cached_config["aws"]["base_s3_uri"],
                    cached_config["aws"]["base_docs_url"]
                ))


            logger.info("doc_url field added to all documents")
            return enriched_documents

        except PresignedUrlGenerationError as e:
            # Handle errors related to presigned URL generation
            logger.error(f"Error generating the presigned URL. {str(e)}.")
            return documents

        except KeyError as err:
            # Handle missing expected document fields
            logger.warning(f"Missing document field. {str(err)} not set.")
            return documents


    @classmethod
    def __add_display_title_to_documents(cls, documents: List[Dict]) -> List[Dict]:
        """
        It adds a default field that can be displayed as document title.

        :param documents: List of Dict objects describing the retrieved chunks.
        :type documents: List[Dict]
        :return: The List of Dict objects describing the retrieved chunksentiched with the new field display_title
        :rtype: List[Dict]
        """
        try:
            enriched_documents = []
            for document in documents:
                document_source = document["_source"]["x-amz-bedrock-kb-source-uri"]
                document["_source"]["display_title"] = document_source.split("/")[-1]
                enriched_documents.append(document)

            logger.info("display_title field added to all documents")
            return enriched_documents
        except KeyError as err:
            # Handle missing parameters and log the error
            logger.warning(f"Missing document field. {str(err)} not set.")
            return documents