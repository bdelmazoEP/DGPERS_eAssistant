# Copyright 2024 European Parliament.
#
# Licensed under the European Union Public License (EUPL), Version 1.2 or subsequent
# versions of the EUPL (the "License"). You may not use this file except in compliance with the License.
# A copy of the License is located at:
# https://eupl.eu/1.2/en/
#
# This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for specific language governing permissions.
import json
import logging
from typing import List, Dict, Any

from ep_nlp.askthedocs_config import askthedocs_config
from ep_nlp.documents_list_formatter import DocumentsListFormatter
from ep_documents_list_formatter.ep_athena_wrapper import EpAthenaThinWrapper
import ep_tools.code_monitor as ep_tools

logger = logging.getLogger(__name__)

class EpDocumentsListFormatter(DocumentsListFormatter):
    """
    A document list formatter that enriches documents with metadata such as URLs, display titles,
    and available linguistic versions from an Athena query.
    """

    def __init__(self, documents: List[Dict[str, Any]]):
        """
        Initializes the formatter with a list of documents.

        Args:
            documents (List[Dict[str, Any]]): A list of document metadata dictionaries.
        """
        self.documents: List[Dict[str, Any]] = documents

    def get_documents(self) -> List[Dict]:
        """Returns the list of formatted documents."""
        return self.documents

    def set_documents(self, documents: List[Dict]) -> None:
        """Sets the list of documents."""
        self.documents = documents

    @ep_tools.measure_performance_decorator(False)
    def format(self) -> None:
        """
        Applies transformations to enrich documents by adding URLs, display titles,
        and linguistic versions.
        """
        self.__add_language_versions()
        self.__add_display_title_to_documents()

    def __add_key_value_to_document_source(self, index: int, key: str, value: Any) -> None:
        """
        Safely adds a key-value pair to the document at the given index.

        Args:
            index (int): Index of the document in the list.
            key (str): Key to be added to the document's _source.
            value (Any): Value to associate with the key.

        Raises:
            IndexError: If the index is out of range.
        """
        if 0 <= index < len(self.documents):
            self.documents[index]["_source"][key] = value
        else:
            raise IndexError("Index out of range")

    def __add_language_versions(self) -> None:
        """
        Enriches each document with available linguistic versions by querying Athena tables.
        The language versions are stored as a list of dictionaries with `lang` and `lv_ffpath`.
        """
        if not self.documents:
            return

        # Extract all document ffpaths in one line
        ffpaths = [doc["_source"]["ffpath"] for doc in self.documents]

        # Query Athena for available linguistic versions
        language_versions = self.__get_language_versions(ffpaths)

        # Assign the retrieved language versions to each document
        self.__set_language_versions(language_versions)

    def __get_language_versions(self, ffpaths: List[str]) -> List[Dict]:
        """
        Queries Athena for linguistic versions based on given file paths.
        It requires the existence of the Athena partitioned table 'ep_all_documents_partitioned'.

        Args:
            ffpaths (List[str]): A list of document file paths.

        Returns:
            List[Dict]: A list of dictionaries containing linguistic version metadata.
        """
        if not ffpaths:
            return []

        sql_set = "(" + ",".join(map(lambda path: f"'{path}'", ffpaths)) + ")"
        sql_query = f"""
            WITH selected_ref AS (
                SELECT fnd, ffpath, codreferencia
                FROM ep_all_documents_partitioned
                WHERE ffpath IN {sql_set}
            )
            SELECT s.ffpath AS document_ffpath, 
                   m.ffpath, 
                   m.longtitle, 
                   m.ididioma
            FROM ep_all_documents_partitioned m 
            JOIN selected_ref s ON m.codreferencia = s.codreferencia  and m.fnd = s.fnd
            ORDER BY s.codreferencia, m.ididioma;
        """

        athena_client = EpAthenaThinWrapper(None)
        logger.info(f"Executing SQL Query: {sql_query}")

        try:
            query_result = athena_client.execute_query(sql_query)

            if query_result is not None:
                response = self.format_lambda_response(query_result["ResultSet"]["Rows"])
                if response["statusCode"] == 200:
                    return response["body"]["records"]
                else:
                    logger.error(f"Athena query failed: {response}")
                    return []
            else:
                return []

        except Exception:
            logger.exception("Error querying Athena")
            return []

    def __set_language_versions(self, lvs: List[Dict]) -> None:
        """
        Assigns language versions to each document.

        Args:
            lvs (List[Dict]): List of language version metadata.
        """

        for index, document in enumerate(self.documents):
            doc_ffpath = document["_source"]["ffpath"]

            # Extract matching language versions for the document
            document_language_versions = [
                {"lang": lv["fields"]["ididioma"], "lv_ffpath": lv["fields"]["ffpath"]}
                for lv in lvs if doc_ffpath == lv["fields"]["document_ffpath"]
            ]
            #
            # If no language version is present -> the documents belongs to "no_clavis" folder.
            # In this case the document is in French.
            if len(document_language_versions) == 0:
                document_language_versions = [{"lang": "fr", "lv_ffpath": doc_ffpath}]
                logger.warning(f"No language versions found for {doc_ffpath}. Assumed French")

            self.__add_key_value_to_document_source(index, "language_versions", document_language_versions)
            logger.info(f"Language versions for document {doc_ffpath} have been added {document_language_versions}")

    def __add_urls_to_documents(self) -> None:
        """
        Enriches documents by replacing their S3 URIs with a public document URL.
        """
        cached_config = askthedocs_config.get_configuration()

        try:
            for document in self.documents:
                document["_source"]["doc_url"] = f"{cached_config['aws']['base_docs_url']}root/{document['_source']['ffpath']}"

            logger.info("doc_url field added to all documents")
        except KeyError as err:
            logger.warning("Missing document metadata ffpath. This is required to use EpDocumentsListFormatter")

    def __add_display_title_to_documents(self) -> None:
        """
        Adds a display title to each document based on the filename extracted from its source URI.
        """
        try:
            for document in self.documents:
                document["_source"]["display_title"] = document["_source"]["x-amz-bedrock-kb-source-uri"].split("/")[-1]

            logger.info("display_title field added to all documents")
        except KeyError as err:
            logger.warning(f"Missing document field. {str(err)} not set.")

    def format_lambda_response(self, athena_query_result: dict) -> Dict:
        """
        Formats athena_query_result in way that is compatible with the aws lambda response.
        Args:
            athena_query_result (): response obtained from a query execution

        Returns:
            formatted results
        """
        headers = athena_query_result[:1][0]
        column_names = []
        for header in headers["Data"]:
            column_names.append(header["VarCharValue"])

        record_counter = 0
        records = []
        for row in athena_query_result[1:]:
            fields = {}
            column_counter = 0
            for value in row["Data"]:
                fields.update({column_names[column_counter]: value["VarCharValue"]})
                column_counter += 1
            record_with_id = {"id": record_counter, "fields": fields}
            records.append(record_with_id)
            record_counter += 1

        return {"statusCode": 200, "body": {"records": records}}