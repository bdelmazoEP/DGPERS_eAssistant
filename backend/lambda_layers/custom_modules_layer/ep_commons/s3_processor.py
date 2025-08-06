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
import hashlib
import logging as log
import os

import boto3
import botocore
import requests
from botocore.exceptions import ClientError
from typing import Iterator, Optional

log.basicConfig(format='%(asctime)s %(message)s')
logger = log.getLogger(__name__)

DEFAULT_DESTINATION_BUCKET = "ep-archives-kendra"

class S3Processor:
    def __init__(self, session: boto3.Session, s3_bucket: str, base_url=None):
        self.s3_client = session.client("s3")
        self.s3_bucket = s3_bucket
        if base_url is None:
            base_url = "https://ep-archives-archibot.s3.eu-central-1.amazonaws.com/"
        self.base_url = base_url

    def get_base_url(self):
        return self.base_url

    def list_objects(self, s3_folder: str=None) -> Iterator[str]:
        """
        Returns the object_keys of the objects present in the s3_folder

        Returns:
            Iterator over the object keys
        """
        if s3_folder is None:
            s3_folder = ""

        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            operation_parameters = {'Bucket': self.s3_bucket,
                                    'Prefix': s3_folder}
            page_iterator = paginator.paginate(**operation_parameters)
        except botocore.exceptions.ClientError as botocore_error:
            logger.error(botocore_error.args[0])
            raise botocore_error
        except botocore.exceptions.ParamValidationError as botocore_error:
            raise ValueError("Incorrect parameter(s): {}".format(botocore_error))

        for page in page_iterator:
            obj_keys = page.get("Contents")
            for obj_key in obj_keys:
                yield obj_key["Key"]


    def download_s3_object(
        self,
        obj_key: str,
        local_folder: str,
        file_name: str = None,
        extra_args: dict = None,
    ):
        """
        Downloads an object from an S3 store
        Args:
            obj_key (str): S3 key of the object to download
            local_folder (str): local folder where downloaded object is saved
            file_name (str): local file name of the downloaded object
            extra_args (dict): extra arguments like ContentType (Content-Type)

        Returns:

        """
        if file_name is None:
            file_name = "to_be_processed.pdf"
        if extra_args is None:
            extra_args = {"ContentType": "text/plain;charset=UTF-8"}
        logger.info(
            ">>>> Downloading {2} {0} {1}...".format(
                obj_key, "{0}/{1}".format(local_folder, file_name), self.s3_bucket
            )
        )

        try:
            print(obj_key, local_folder, file_name)
            self.s3_client.download_file(
                self.s3_bucket,
                obj_key,
                "{0}/{1}".format(local_folder, file_name, ExtraArgs=extra_args),
            )
        except botocore.exceptions.ClientError as botocore_error:
            logger.error(botocore_error.args[0])
            raise botocore_error

        except botocore.exceptions.ParamValidationError as botocore_error:
            raise ValueError("Incorrect parameter(s): {}".format(botocore_error))

        logger.info(
            ">>>> Downloading {2} {0} {1} successfully completed.".format(
                obj_key, "{0}/{1}".format(local_folder, file_name), self.s3_bucket
            )
        )

    def save_text_to_s3_object(self, obj_key: str, text: str, destination_bucket=None, content_type=None):
        """
        Saves the provided text to a file in an S3 bucket.

        This method uploads the given text as a file to the specified S3 bucket using the provided
        S3 key. Optional parameters allow specifying a different destination bucket and content type.

        Args:
            obj_key (str): Object key of the text file to be created in S3.
            text (str): Text to be saved in the S3 file.
            destination_bucket (Optional[str]): Bucket that will store the created file. 
                                                Defaults to `DEFAULT_DESTINATION_BUCKET`.
            content_type (Optional[str]): Content type of the text file. 
                                          Defaults to 'text/plain;charset=UTF-8'.

        Returns:
            None
        """
        if destination_bucket is None:
            destination_bucket = DEFAULT_DESTINATION_BUCKET
        if content_type is None:
            content_type = "text/plain;charset=UTF-8"

        try:
            self.s3_client.put_object(
                Body=text,
                Bucket=destination_bucket,
                ContentType=content_type,
                Key=obj_key,
            )
        except botocore.exceptions.ClientError as botocore_error:
            logger.error(botocore_error.args[0])
            raise botocore_error
        except botocore.exceptions.ParamValidationError as botocore_error:
            raise ValueError("Incorrect parameter(s): {}".format(botocore_error))

    def upload_s3_object(
        self,
        obj_key: str,
        local_folder: Optional[str],
        file_name: Optional[str]=None,
        destination_bucket: Optional[str]=None,
        extra_args: Optional[dict]=None,
        delete_local=False
    ):
        """
        Uploads a local file to an S3 bucket.

        This method uploads a file from a specified local folder to an S3 bucket using the provided
        S3 key. Optional parameters allow specifying a different file name, bucket, extra arguments
        for the upload, and whether to delete the local file after uploading.

        Args:
            obj_key (str): S3 key of the object when uploaded.
            local_folder (str): Folder containing the file to be uploaded.
            file_name (Optional[str]): Name of the file to be uploaded. Defaults to 'to_be_processed.pdf'.
            destination_bucket (Optional[str]): Name of the bucket where the file must be uploaded.
                                                Defaults to the instance's S3 bucket.
            extra_args (Optional[dict]): Extra arguments for the upload, such as ContentType.
                                         Defaults to {'ContentType': 'text/plain;charset=UTF-8'}.
            delete_local (bool): If True, the local file is removed after uploading. Defaults to False.

        Returns:
            None
        """

        if file_name is None:
            file_name = "to_be_processed.pdf"
        if destination_bucket is None:
            destination_bucket = self.s3_bucket
        if extra_args is None:
            extra_args = {"ContentType": "text/plain;charset=UTF-8"}

        logger.info(
            "{2} {0} {1}".format(
                obj_key, "{0}/{1}".format(local_folder, file_name), self.s3_bucket
            )
        )

        try:
            self.s3_client.upload_file(
                "{0}/{1}".format(local_folder, file_name),
                destination_bucket,
                obj_key,
                ExtraArgs=extra_args,
            )
        except botocore.exceptions.ClientError as botocore_error:
            logger.error(botocore_error.args[0])
            raise botocore_error

        except botocore.exceptions.ParamValidationError as botocore_error:
            raise ValueError("Incorrect parameter(s): {}".format(botocore_error))

        if delete_local:
            os.remove("{0}/{1}".format(local_folder, file_name))

        logger.info(
            "Uploading {2} {0} {1} successfully completed.".format(
                obj_key, "{0}/{1}".format(local_folder, file_name), self.s3_bucket
            )
        )

    def delete_s3_object(self, obj_key: str) -> Optional[str]:
        """
        Deletes an object from the S3 bucket.

        This method deletes an object specified by the S3 key from the instance's S3 bucket. It first
        checks if the object exists in the bucket. If the object does not exist, it logs this information
        and returns None. If the object exists, it attempts to delete the object and returns the key of
        the deleted object.

        Args:
            obj_key (str): Key of the object to be deleted.

        Returns:
            Optional[str]: None if the object doesn't exist, otherwise the key of the deleted object.
        """
        if not self.check_existence_s3_object(obj_key, self.s3_bucket):
            logger.info("Object {0} does not exists in bucket {1}.".format(obj_key, self.s3_bucket))
            return None

        try:
            self.s3_client.delete_object(Key=obj_key, Bucket=self.s3_bucket)
        except botocore.exceptions.ClientError as botocore_error:
            logger.error(botocore_error.args[0])
            raise botocore_error
        except botocore.exceptions.ParamValidationError as botocore_error:
            raise ValueError("Incorrect parameter(s): {}".format(botocore_error))

        return obj_key

    def compute_checksumMD5(self, file_path: str) -> Optional[str]:
        '''
        It computes the md5 value of the file in input.
        Args:
            file_path (str): path of the file whose md5 value must be computed.

        Returns:
            str : the computed md5 value
        '''

        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)

        return hash_md5.hexdigest()


    def get_s3_checksumMD5(self, obj_key: str) -> Optional[str]:
        '''
        It gets the ETag value of the object identified by obj_key.
        Args:
            obj_key (str): S3 key of the object

        Returns:
            str : the md5 value
        '''

        md5 = None
        try:
            response = self.s3_client.head_object(Bucket=self.s3_bucket, Key=obj_key)
            md5 = response['ETag']
        except ClientError as e:
            error = e.response["Error"]["Code"]
            logger.error(f"Error={error} retrieving file {obj_key}")

        #
        # header field for ETag defines the quotes as part of entity-tag value. Therefore the returned md5 value must be
        # unquoted. See: https://github.com/aws/aws-sdk-go-v2/issues/1696
        return md5.replace("\"", "")

    def get_s3_text_file_content_by_url(
        self,
        file_name,
        url=None
    ) -> Optional[str]:
        """
        It extracts the text from a file in the Internet.
        Args:
            file_name (str): Name of the file
            url (str): base url of the server publishing the file

        Returns:

        """
        if url is None:
            url = "https://ep-archives-archibot.s3.eu-central-1.amazonaws.com/root/"

        try:
            response = requests.get(
                "{0}{1}.full.txt".format(url, file_name), verify=False
            )
        except requests.exceptions.HTTPError as errh:
            logger.error(errh.args[0])
            return None

        logger.info(
            "File {0}{1}: status_code={2}".format(url, file_name, response.status_code)
        )
        if response.status_code == 200:
            return response.text
        else:
            return None

    def check_existence_s3_object(self, obj_key: str, bucket_name: str = None) -> bool:
        """
        Checks the existence of an S3 object.

        Args:
            obj_key (str): S3 key of the object to be checked
            bucket_name (str):  name of the bucket where the file must be checked

        Returns:

        """
        if bucket_name is None:
            bucket_name = self.s3_bucket

        logger.info(
            ">>>> Checking existence {0} {1}...".format(obj_key, bucket_name)
        )

        try:
            self.s3_client.head_object(Bucket=bucket_name, Key=obj_key)
            logger.info(
                "Object {0} exists in s3 bucket {1}".format(obj_key, bucket_name)
            )
            return True
        except:
            logger.info(
                "Object {0} doesn't exist in s3 bucket {1}".format(obj_key, bucket_name)
            )
            return False

    def get_s3_text_file_content(self, obj_key: str, bucket=None) -> Optional[str]:
        """Retrieves the text content of a text file stored in s3.

        :param str obj_key: object key
        :param str bucket: bucket where to search the file

        :returns:  string (unicode) containing the extracted text or an empty string if the pdf is empty or raster

        """
        if bucket is None:
            bucket = self.s3_bucket
        else:
            bucket = bucket

        try:
            s3_object = self.s3_client.get_object(Bucket=bucket, Key=obj_key)
            logger.debug(f"Retrieved file {obj_key}")
            return s3_object["Body"].read().decode("utf-8")
        except ClientError as e:
            error = e.response["Error"]["Code"]
            logger.error(f"Error={error} retrieving file {obj_key}")

    def get_s3_text_file_content_list(self, obj_keys: list[str]) -> Iterator[str]:
        """
        Retrieves the text content of a list of files stored in s3.
        Args:
            obj_keys (list(str)): list of obj_keys

        Returns:
            list of strings containing the files contents
        """
        for obj_key in obj_keys:
            yield obj_key, self.get_s3_text_file_content(obj_key)

    def set_bucket(self, s3_bucket: str):
        self.s3_bucket = s3_bucket

    def copy_object(self, departure_bucket: str, destination_bucket: str, dep_obj_key: str, dest_obj_key:str) -> bool:
        '''
        Copies an object from a bucket to another
        Args:
            departure_bucket ():
            destination_bucket ():
            dep_obj_key ():
            dest_obj_key ():

        Returns:

        '''
        #
        # Check dep_obj_key existence
        copy_source = {
            'Bucket': departure_bucket,
            'Key': dep_obj_key
        }

        try:
            response = self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=destination_bucket,
                Key=dest_obj_key
            )
        except botocore.exceptions.ClientError as botocore_error:
            logger.error(botocore_error.args[0])
            raise botocore_error
        except botocore.exceptions.ParamValidationError as botocore_error:
            raise ValueError("Incorrect parameter(s): {}".format(botocore_error))

        logger.info("{0} {1} {2} {3}".format(departure_bucket, destination_bucket, dep_obj_key, dest_obj_key))
        print(response)
        return True