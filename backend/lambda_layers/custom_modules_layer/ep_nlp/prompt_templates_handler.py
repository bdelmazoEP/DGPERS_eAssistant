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
import os
from io import StringIO
from typing import Optional

import boto3
import botocore
from botocore.exceptions import ClientError
import pandas as pd
from ep_nlp.askthedocs_config import askthedocs_config

logger = logging.getLogger(__name__)

CONFIG_BUCKET = os.getenv("askthedocs_config_bucket")

def __read_csv__() -> Optional[pd.DataFrame]:
    """
    Reads a CSV file from an S3 bucket and loads it into a pandas DataFrame.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the CSV data.
    """
    cached_config = askthedocs_config.get_configuration()
    try:
        s3_client = boto3.client('s3')
        # If the environmant variable is not set, attempt to retrieve its value from the configuration file.
        prompt_templates_table_key = os.getenv("ai_prompt_table_key", cached_config["ai"]["prompt_table_key"])

        csv_obj = s3_client.get_object(Bucket=CONFIG_BUCKET,
                                       Key=prompt_templates_table_key)
        csv_data = csv_obj['Body'].read().decode('utf-8')
        data = StringIO(csv_data)
        logger.info(f"Prompt templates successfully read from {prompt_templates_table_key}")
        return pd.read_csv(data, sep=";", encoding="utf-8")
    except botocore.exceptions.ClientError as botocore_error:
        logger.error(botocore_error.args[0])
        raise botocore_error
    except botocore.exceptions.ParamValidationError as botocore_error:
        raise ValueError("Incorrect parameter(s): {}".format(botocore_error))


class PromptTemplatesHandlerMeta(type):

    _instances = {}  # Dictionary to store the single instance

    def __call__(cls, *args, **kwargs):
        # If the class instance is not yet created, create it
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        # Return the existing instance, ensuring only one instance is created (singleton)
        return cls._instances[cls]

class PromptTemplatesHandler(metaclass=PromptTemplatesHandlerMeta):

    def __init__(self):
        # Any attributes or setup you need for the singleton
        self.prompt_templates: pd.DataFrame = None

    def set_prompt_templates(self, prompt_templates: pd.DataFrame):
        self.prompt_templates = prompt_templates

    def get_prompt_templates(self) -> pd.DataFrame:
        return self.prompt_templates

# askthedocs_prompt_templates = PromptTemplatesHandler()

# askthedocs_prompt_templates.set_prompt_templates(__read_csv__())
