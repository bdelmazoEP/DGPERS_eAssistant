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
from typing import List, Dict
import botocore
import botocore.exceptions
import numpy as np
from ep_nlp.utils import initialize_client

logger = logging.getLogger(__name__)


class Text2VectorProcessor:
    """
    Class for processing text and generating vector embeddings using AWS Bedrock.
    """

    def __init__(self, aws_region: str, model_id: str = None):

        """
        Initialize the EpText2VectorProcessor with a specific model ID.

        Args:
            aws_region (str): id of the aws region from where the model is used
            model_id (str): The identifier of the model to be used for text embedding.
        """
        if model_id is None:
            model_id = "amazon.titan-embed-text-v2:0"

        self.model_id = model_id
        self.aws_region = aws_region
        # Initialize the Bedrock runtime client
        self.bedrock_client = initialize_client("bedrock-runtime", self.aws_region)

    def invoke_titan_model(self, text: str):
        response = self.bedrock_client.invoke_model(body=json.dumps({"inputText": text}),
                                                    modelId=self.model_id,
                                                    accept='application/json',
                                                    contentType='application/json')
        return response

    def invoke_cohere_model(self, text: str):
        response = self.bedrock_client.invoke_model(body=json.dumps({"texts": [text], "input_type": "search_document"}),
                                                    modelId=self.model_id,
                                                    accept='*/*',
                                                    contentType='application/json')
        return response

    def extract_embeddings(self, response: Dict):
        if "titan" in self.model_id:
            return response.get('embedding')
        elif "cohere" in self.model_id:
            return np.array(response['embeddings'][0])

        return None

    def get_embeddings(self, text: str) -> List[float]:
        """
        Generate vector embeddings for the input text using the specified model.
        See also: https://aws.amazon.com/blogs/machine-learning/getting-started-with-amazon-titan-text-embeddings/
        Args:
            text (str): The input text to generate embeddings for.

        Returns:
            List[float]: The vector embeddings for the input text.
        """
        # Invoke the Bedrock model to generate embeddings
        try:
            if "titan" in self.model_id:
                response = self.invoke_titan_model(text)
            elif "cohere" in self.model_id:
                response = self.invoke_cohere_model(text)
            else:
                logger.warning(f"Model {self.model_id} is not supported")
                return []

        except botocore.exceptions.ClientError as error:
            # Handle AWS client errors
            raise error
        except botocore.exceptions.ParamValidationError as error:
            # Handle parameter validation errors
            raise ValueError('The parameters you provided are incorrect: {}'.format(error))

        # Parse the response and extract embeddings
        result = json.loads(response['body'].read())

        return self.extract_embeddings(result)
