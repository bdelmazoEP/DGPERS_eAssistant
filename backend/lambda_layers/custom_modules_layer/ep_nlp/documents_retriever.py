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
from abc import abstractmethod
from typing import List, Dict, Optional

from ep_nlp.query_analyzer import QueryElements


class DocumentsRetriever:
    """
    Class used to interact with Amazon OpenSearch Service.
    """

    def get_retriever_type(self) -> str:
        return "opensearch"

    @abstractmethod
    def retrieve_context(self, query: str, **kwargs) -> List[Dict]:
        pass

    @classmethod
    def isPresent(cls) -> bool:
        return True
