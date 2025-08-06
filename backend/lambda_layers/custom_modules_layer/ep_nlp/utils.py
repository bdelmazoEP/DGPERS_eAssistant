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

import importlib
import logging
from typing import Dict, Optional
import pandas as pd
import boto3
import botocore
from botocore.client import BaseClient
import botocore.exceptions

logger = logging.getLogger(__name__)

from ep_nlp.askthedocs_config import askthedocs_config
# from ep_nlp.prompt_templates_handler import askthedocs_prompt_templates ########## removed import as handler class reading the CSV no longer needed

##########
import json
##########


cached_config: Dict = None
cached_prompt_table: pd.DataFrame = None

class UtilsError(Exception):
    "Custom exception for errors returned by EpPromptManager"

    def __init__(self, message):
        self.message = message


def import_class(module_name: str, class_name: str) -> type:
    """
    Dynamically imports a class from a specified module.

    Args:
        module_name (str): The name of the module to import.
        class_name (str): The name of the class to retrieve from the module.

    Returns:
        type: The class object specified by class_name from the module.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the class is not found in the module.
    """
    try:
        module = importlib.import_module(module_name, class_name)
        class_obj = getattr(module, class_name)
        return class_obj
    except ImportError as error:
        logger.error(f"The module {module_name} cannot be imported. {error.name}")
        raise UtilsError(f"The module {module_name} cannot be imported. {error.name}")
    except AttributeError as error:
        logger.error(f"The class {class_name} could not be found in the module. {error}")
        raise UtilsError(f"The class {class_name} could not be found in the module. {error}")


def get_request_body_template(llm_model_id: str, template_name: str) -> Optional[Dict]:
    """
    Retrieves a request body template from the database based on the specified language model and template type.

    Args:
        llm_model_id (str): The identifier for the language model.
        template_type (str): The type of prompt template to retrieve.

    Returns:
        dict: A dictionary containing the template details including instructions, prompt, max tokens, and temperature.

    Raises:
        EpChatbotError: If the specified prompt template is not found in the database.

    """
    cached_config = askthedocs_config.get_configuration()
    # cached_prompt_table = askthedocs_prompt_templates.get_prompt_templates() ########## removed handler

    if cached_config is None:
        logger.error("No configuration found. It is impossible to read the prompt table values.")
        return None

    ########## removed handler vvv
    # if cached_prompt_table is None:
    #     logger.error("No prompt templates list found. It is impossible to read the prompt table values.")
    #     return None

    # result = cached_prompt_table.query('MODEL=="{0}" and NAME=="{1}"'.format(llm_model_id, template_name))

    # if len(result) < 1:
    #     logger.error(f"Prompt template {template_name} not found for model {llm_model_id}")
    #     raise UtilsError(f"Prompt template {template_name} not found for model {llm_model_id}")

    # logger.debug(result)
    # return {"SYSTEM": result.iloc[0]["SYSTEM"],
    #         "REQUEST_BODY": result.iloc[0]["REQUEST_BODY"],
    #         "PROMPT": result.iloc[0]["PROMPT"]}
    ########## ^^^

    ########## new prompt logic:
    # logger.debug('AAA11: using new prompt importing logic') ####################
    return {"SYSTEM": cached_config['prompt_templates'][template_name]['system'],
            "REQUEST_BODY": cached_config['prompt_templates'][template_name]['request'], 
            "PROMPT": cached_config['prompt_templates'][template_name]['prompt']}
    ##########




def remove_non_printable(input_string):
    """
    Removes all non-printable characters from a given string.

    Parameters:
        input_string (str): The string to be cleaned.

    Returns:
        str: A new string containing only printable characters.

    Notes:
        - A character is considered printable if `char.isprintable()` returns True.
        - Printable characters include letters, digits, punctuation, and whitespace.
        - Non-printable characters, such as control characters (e.g., '\x00', '\x1F'),
          are excluded from the output.
    """
    return ''.join(char for char in input_string if char.isprintable())

def initialize_client(client_type: str, aws_region: str) -> BaseClient:
    """
    Initialize an AWS service client using boto3.

    Args:
        client_type (str): The type of AWS service client to initialize (e.g., 'bedrock-runtime').

    Returns:
        BaseClient: A boto3 client for the specified AWS service.

    Raises:
        ClientError: If there is an error with the AWS service client.
        ValueError: If the provided parameters are incorrect.
        :param aws_region: aws region
        :type aws_region:
    """

    try:
        # If the environmant variable is not set, attempt to retrieve its value from the configuration file.
        client = boto3.client(client_type, region_name=aws_region)
    except botocore.exceptions.ClientError as error:
        # Handle AWS client errors
        raise error
    except botocore.exceptions.ParamValidationError as error:
        # Handle parameter validation errors
        raise ValueError('The parameters you provided are incorrect: {}'.format(error))

    return client


