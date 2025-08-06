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
from pathlib import Path
# from typing import Dict

import boto3

logger = logging.getLogger(__name__)

################################
CONFIG_PATH = Path(os.getenv("config_file_path")) ################################
################################

# metaclass to make it a singleton
class AskthedocsConfigMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class AskthedocsConfig(metaclass=AskthedocsConfigMeta):
    def __init__(self):
        self.configuration = self._load_configuration()

    def _load_configuration(self) -> dict:
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                logger.info(f"Loading configuration from {CONFIG_PATH}")
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return {}

    def get_configuration(self) -> dict:
        return self.configuration


# Singleton instance
askthedocs_config = AskthedocsConfig()



# CONFIG_FILE = os.getenv("askthedocs_config_file")
# CONFIG_BUCKET = os.getenv("askthedocs_config_bucket")

# def upload_configuration(config_bucket: str, config_file: str) -> Dict:
#     """
#     Fetches the configuration from an S3 bucket.

#     Parameters:
#         config_bucket (str): The name of the S3 bucket.
#         config_file (str): The key (file name) of the configuration file in the S3 bucket.

#     Returns:
#         Dict: The configuration data loaded as a dictionary.
#     """

#     print(f"Fetching configuration from S3...{config_bucket} {config_file}")

#     s3 = boto3.client('s3')
#     response = s3.get_object(Bucket=config_bucket, Key=config_file)

#     return json.loads(response['Body'].read())


# class AskthedocsConfigMeta(type):

#     _instances = {}  # Dictionary to store the single instance

#     def __call__(cls, *args, **kwargs):
#         # If the class instance is not yet created, create it
#         if cls not in cls._instances:
#             instance = super().__call__(*args, **kwargs)
#             cls._instances[cls] = instance
#         # Return the existing instance, ensuring only one instance is created (singleton)
#         return cls._instances[cls]

# # Using the metaclass
# class AskthedocsConfig(metaclass=AskthedocsConfigMeta):

#     def __init__(self):
#         # Any attributes or setup you need for the singleton
#         self.configuration = {}

#     def set_configuration(self, configuration: Dict):
#         self.configuration = configuration

#     def get_configuration(self):
#         return self.configuration

# askthedocs_config = AskthedocsConfig()
# askthedocs_config.set_configuration(upload_configuration(CONFIG_BUCKET, CONFIG_FILE))
