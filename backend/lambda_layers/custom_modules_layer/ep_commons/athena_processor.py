import io
import os
import time
from typing import Optional, List

import pandas as pd
import logging
import json

import boto3
import botocore.exceptions
# import awswrangler as wr

class EpAthenaHelper:
    def __init__(
        self,
        session,
        database="athenadb",
        region_name="eu-central-1",
        s3_bucket="ep-archives-archibot",
        temporary_files_folder="athena",
    ):
        if not session:
            session = boto3.Session(region_name=region_name)

        self.session = session
        self.athena_client = session.client("athena")
        self.database = database
        self.s3_client = session.client("s3")
        self.s3_bucket = s3_bucket
        self.aws_region = region_name
        self.temporary_files_folder = temporary_files_folder
        self.logger = logging.getLogger(__name__)

    def create_named_query(self, query_name, sql_query, workgroup="primary"):
        try:
            response = self.athena_client.create_named_query(
                Name=query_name,
                Database=self.database,
                QueryString=sql_query,
                WorkGroup=workgroup,
            )
            return response

        except botocore.exceptions.ClientError as error:
            self.logger.debug("Method failed with error {0}".format(error))
            raise error

        return None

    def list_named_queries(self, workgroup="primary", maxresult=10):
        """
        It returns ta list of queries saved by the specified workgroup.
        Args:
            workgroup (str): The workgroup owning the list of queries
            maxresult (int): Max size of the returned list

        Returns:
            Response containing the list of found queries
        """
        try:
            response = self.athena_client.list_named_queries(
                MaxResults=maxresult, WorkGroup=workgroup
            )
            return response

        except botocore.exceptions.ClientError as error:
            self.logger.debug("Method failed with error {0}".format(error))
            raise error

    def get_named_query_by_id(self, id: str):
        """
        It returns the saved query with the specified id.

        Args:
            id (str): d of the searched query

        Returns: The query with the given id. None if the query doesn't exist.

        """
        try:
            response = self.athena_client.get_named_query(NamedQueryId=id)
            return response

        except botocore.exceptions.ClientError as error:
            print(error)
            raise error

    def get_named_queries_by_name(self, name: str, workgroup="primary") -> list():
        try:
            response = self.athena_client.list_named_queries(WorkGroup=workgroup)
            query_ids = response["NamedQueryIds"]
            result = []
            for id in query_ids:
                r = self.get_named_query_by_id(id)
                if r["NamedQuery"]["Name"] == name:
                    result.append(r)

            return result

        except botocore.exceptions.ClientError as error:
            self.logger.error(error)
            raise error

    def delete_named_query_by_id(self, id: str) -> str:
        try:
            self.athena_client.get_named_query(NamedQueryId=id)

        except self.athena_client.exceptions.InvalidRequestException as e:
            self.logger.warning(f"Named query id {id} doesn't exist.")
            return None

        try:
            self.athena_client.delete_named_query(NamedQueryId=id)
            return id

        except botocore.exceptions.ClientError as error:
            self.logger.error(error)
            raise error

    def __get_query_execution_id(self, sql: str) -> Optional[str]:
        try:
            response = self.athena_client.start_query_execution(
                QueryString=sql,
                QueryExecutionContext={"Database": self.database},
                ResultConfiguration={
                    "OutputLocation": "s3://{0}/{1}/".format(
                        self.s3_bucket, self.temporary_files_folder
                    ),
                },
            )
            self.logger.info(
                "Execution started with id={0}".format(response["QueryExecutionId"])
            )
        except self.athena_client.exceptions.InvalidRequestException as e:
            self.logger.error(
                f'Issues starting query {sql} - {e.response["Error"]["Code"]}'
            )
            return None
        except self.athena_client.exceptions.InternalServerException as e:
            self.logger.error(
                f'Issues starting query {sql} - {e.response["Error"]["Code"]}'
            )
            return None

        return response["QueryExecutionId"]

    def __clean_query_temporary_files(self, execution_id):
        obj_key = "{0}/{1}.csv".format(self.temporary_files_folder, execution_id)
        self.__delete_s3_object(obj_key)
        obj_key = "{0}.metadata".format(obj_key)
        self.__delete_s3_object(obj_key)
        self.logger.info(
            "Deleted temporary files of execution with id={0}".format(execution_id)
        )

    def __get_query_results(self, execution_id: str) -> dict:
        self.logger.debug("id_execution={0}".format(execution_id))
        response = self.athena_client.get_query_results(QueryExecutionId=execution_id)
        self.__clean_query_temporary_files(execution_id)
        self.logger.info("Query successfully completed.")

        return response

    def __get_large_query_results(self, execution_id: str) -> pd.DataFrame:
        """
        Fetches large amount of data retrieved by a query. Instead of using the athena client method
        get_query_results, that requires to paginate the query results, it reads the temporary csv file
        saved by the execution process in s3. Before leaving, the methods deletes the temporary files.

        Args:
            execution_id (str):

        Returns:
            A dataframe with the query results
        """
        try:
            obj_key = "{0}/{1}.csv".format(self.temporary_files_folder, execution_id)
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=obj_key)
            rows = pd.read_csv(io.BytesIO(response["Body"].read()), encoding="utf8")
            self.__clean_query_temporary_files(execution_id)
            self.logger.info(
                "Large query successfully completed. Returned {0} rows".format(
                    len(rows)
                )
            )

            return rows

        except Exception as e:
            self.logger.info(e)

    def __wait_for_query_completion(
        self, execution_id: str, repeat=15, interval=5
    ) -> str:
        """
        Rudimentary method that simplifies te use of asynchronous methods for query execution.

        Args:
            execution_id (str): id of the running job
            repeat (int): how many loops must be performed waiting for the query result
            interval (int): how many seconds between two loops

        Returns:

        """
        state = "RUNNING"

        try:
            ## https: // docs.aws.amazon.com / athena / latest / APIReference / API_QueryExecutionStatus.html
            while repeat >= 0 and state in ["RUNNING", "QUEUED", "FAILED"]:
                repeat -= 1

                response = self.athena_client.get_query_execution(
                    QueryExecutionId=execution_id
                )
                if (
                    "QueryExecution" in response
                    and "Status" in response["QueryExecution"]
                    and "State" in response["QueryExecution"]["Status"]
                ):
                    state = response["QueryExecution"]["Status"]["State"]
                    self.logger.debug("state={0}, counter={1}".format(state, repeat))
                    if state == "SUCCEEDED":
                        return state

                time.sleep(interval)
            ## End loop

        except self.athena_client.exceptions.InvalidRequestException as e:
            self.logger.error(
                f'Issues executing query {execution_id} - {e.response["Error"]["Code"]}'
            )
            return None

        return None

    def __delete_s3_object(self, obj_key):
        """
        Checks if the object idenified by obj_key exists and, if yes, it deletes it.

        Args:
            obj_key (): key of s3 object to be deleted

        Returns:
            http status code
        """
        try:
            self.s3_client.head_object(Bucket=self.s3_bucket, Key=obj_key)
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return "404"

        response = self.s3_client.delete_object(Bucket=self.s3_bucket, Key=obj_key)

        return response["ResponseMetadata"]["HTTPStatusCode"]

    def format_lambda_response(self, athena_query_result: dict) -> {}:
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

    def __start_and_wait_for_query_execution(self, sql: str, repeat=12, interval=5, execution_id=None) -> str:
        if execution_id is None:
            execution_id = self.__get_query_execution_id(sql)
        state = self.__wait_for_query_completion(execution_id, repeat, interval)
        return state

    def execute_large_query(self, sql: str, repeat=12, interval=5) -> pd.DataFrame:
        """
        Wrapper method that simplifies the use of asynchronous aws athena methods. This method returns an arbitrary number
        of rows.
        Args:
            sql (str): string containing the sql query
            repeat (int): how many loops must be performed waiting for the query result
            interval (int): how many seconds between two loops

        Returns:
            A pandas DataFrame containing the query result
        """

        execution_id = self.__get_query_execution_id(sql)
        if "SUCCEEDED" == self.__start_and_wait_for_query_execution(
            sql, repeat, interval
        ):
            return self.__get_large_query_results(execution_id)

        return None

    def execute_query(self, sql: str, repeat=12, interval=5) -> {}:
        """
        Wrapper method that simplifies the use of asynchronous aws athena methods. It returns max 999 rows.
        Args:
            sql (str): string containing the sql query
            repeat (int): how many loops must be performed waiting for the query result
            interval (int): how many seconds between two loops

        Returns:
            A dict containing the query result
        """
        execution_id = self.__get_query_execution_id(sql)
        if "SUCCEEDED" == self.__start_and_wait_for_query_execution(
            sql, repeat, interval, execution_id
        ):
            return self.__get_query_results(execution_id)

        return None

    def execute_wr_query(self, sql: str) -> pd.DataFrame:
        """
        Wrapper method that executes a query against an AWS Athena db using AWS Data Wrangler.
        Args:
            sql (str): string containing the sql query

        Returns:
            A dict containing the query result
        """

        df = wr.athena.read_sql_query(sql, database=self.database, ctas_approach=False, boto3_session=self.session)

        return df


    def get_response(self, sql_query: str) -> {}:
        """
        Performs a query and returns a response ready to be used by a aws lambda function.
        Args:
            sql_query (str):

        Returns:
            A dict containing the query result

        """
        query_result = self.execute_query(sql_query)

        if query_result is not None:
            response = self.format_lambda_response(query_result["ResultSet"]["Rows"])
        else:
            response = {
                "statusCode": 400,
                "body": json.dumps({"msg": "Query was not successful."}),
            }

        return response

    def get_document_language_codes(self, event, context: Optional[dict]) -> {}:
        """
        Helper method that performs the search of documents given their dossierName.

        Args:
            event (dict): dict containing the dossierName to be used for the search.
            context (Optional[dict]): currently not used.

        Returns:
            A dict with the result of the query
        """

        fileName = event["queryStringParameters"]["fileName"]
        sql_query = f"SELECT CODREFERENCIA, IDIDIOMA FROM myview where TYPE='DOC' and DOS='{fileName}'"
        response = self.get_response(sql_query)

        return response

    def get_series(self, event, context):
        """
        Helper method that performs the search of series (files) given their fondsName.

        Args:
            event (dict): disct containing the fondsName to be used for the search.
            context (Optional[dict]): currently not used.

        Returns:
            A dict with the result of the query
        """

        fondsName = event["queryStringParameters"]["fondsName"]
        sql_query = f"SELECT CODREFERENCIA, TITULO, CODREFFONDO, TYPE, BEGINDATE, ENDDATE FROM \"myview\" where TYPE='SER' and TIPO='4' and CODREFERENCIA like '{fondsName}%' order by CODREFERENCIA"
        response = self.get_response(sql_query)

        return response

    def get_files_reference_codes(self, event, context: Optional[dict]):
        """
        Helper method that performs the search of dossiers (files) given their serieName and year.

        Args:
            event (dict): disct containing the serieName and year to be used for the search.
            context (Optional[dict]): currently not used.

        Returns:
            A dict with the result of the query
        """

        serieName = event["queryStringParameters"]["serieName"]
        year = event["queryStringParameters"]["year"]
        sql_query = f"SELECT CODREFERENCIA, TITULO, TYPE, EXTRACT(YEAR from from_iso8601_date(BEGINDATE)) as YEAR FROM myview where TYPE='DOS' and DOS LIKE '{serieName}%' and BEGINDATE IS NOT NULL and EXTRACT(YEAR from from_iso8601_date(BEGINDATE))={year}"
        response = self.get_response(sql_query)

        return response

    def get_documents_reference_codes(self, event: {}, context: Optional[dict]) -> {}:
        """
        Helper method that performs the search of documents given their dossierName and epLanguage.

        Args:
            event (dict): disct containing the dossierName and epLanguage to be used for the search.
            context (Optional[dict]): currently not used.

        Returns:
            A dict with the result of the query
        """

        dossier_name = event["queryStringParameters"]["dossierName"]
        ep_language = event["queryStringParameters"]["epLanguage"]
        sql_query = f"select CODREFERENCIA, TITULO, IDIDIOMA,FFPATH from myview where DOS='{dossier_name}' and IDIDIOMA='{ep_language}'"
        response = self.get_response(sql_query)

        return response

    def get_document_metadata_by_ffpath(self, event: dict) -> Optional[dict]:
        """

        Args:
            ffpath (str): FFPATH value of the document whose metadata are searched for.

        Returns:
            A dict with the result of the query
        """
        ffpath = event["queryStringParameters"]["ffpath"].upper()
        sql_query = f"select CODREFERENCIA, TITULO, LONGTITLE, IDIDIOMA, FFPATH from myview where upper(ffpath) = '{ffpath}'"
        response = self.get_response(sql_query)

        return response

    def get_documents_reference_codes_by_longtitle(
        self, event: {}, context: Optional[dict]
    ) -> {}:
        """
        Helper method that performs the search of dossiers (files) given their longtitle.
        Args:
            event (dict): disct containing the longtitle to be used for the search.
            context (Optional[dict]): currently not used.

        Returns:
            A dict with the result of the query
        """
        longtitle = event["queryStringParameters"]["longtitle"].upper()
        sql_query = f"select CODREFERENCIA, TITULO, LONGTITLE, IDIDIOMA, FFPATH from myview where ((upper(LONGTITLE) like '{longtitle}') or (strpos(LONGTITLE, '{longtitle}') > 0)) and TYPE='DOS'"
        response = self.get_response(sql_query)

        return response

    def execute_named_query(self, event: dict, context: Optional[dict]) -> pd.DataFrame:
        """
            Performs the query whose name is in event. If the query doesn't exist, it returns None.
        Args:
            event (dict): the dict that contains the query name
            context (Optional[dict]): not used

        Returns:
            A pandas dataframe with the result rows
        """
        query_name = event["queryStringParameters"]["queryName"]
        self.logger.debug("Name of the query to be processed: {0}".format(query_name))

        queries = self.get_named_queries_by_name(query_name)
        if len(queries) > 0:
            sql_query = queries[0]["NamedQuery"]["QueryString"]
            self.logger.debug("Query to be processed: {0}".format(sql_query))
            df = self.execute_large_query(sql_query)
        else:
            df = None

        return df

