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
from typing import Dict, List

class DocumentsListFormatter():

    @abstractmethod
    def format(self) -> List[Dict]:
        """
        :return: the formatted list of documents
        """
        pass

    @abstractmethod
    def get_documents(self) -> List[Dict]:
        pass

    @abstractmethod
    def set_documents(self, documents_list: List[Dict]):
        pass