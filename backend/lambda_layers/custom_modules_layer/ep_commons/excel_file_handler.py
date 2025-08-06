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
from typing import List, Optional

import pandas as pd

from os import path

log = logging.getLogger("ep_excel_file_handler")
log.setLevel(logging.INFO)

# Create handlers
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.ERROR)

# Create formatters and add it to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
log.addHandler(c_handler)


class ExcelFileHandler:

    def __init__(self, excel_path: str=None, df: pd.DataFrame=None, sheet_name: str=None, sep: str=None):

        if excel_path is None:
            self.xslx_content = df
        else:
            if sheet_name is None:
                if excel_path.split(".")[-1] == "csv":
                    if sep is None:
                        sep = ','
                    self.xslx_content = ExcelFileHandler.read_csv(excel_path, sep=sep)
                else:
                    self.xslx_content = ExcelFileHandler.read_excel(excel_path)
            else:
                self.xslx_content = ExcelFileHandler.read_excel(excel_path, sheet_name=sheet_name)


    @staticmethod
    def read_csv(csv_path: str, sep=None) -> Optional[pd.DataFrame]:

        if sep is None:
            sep = ";"

        if path.exists(csv_path):
            df = pd.read_csv(csv_path, sep=sep, engine="python")
            return df
        else:
            log.warning('File {0} doesn\'t exists.'.format(csv_path))
            return None


    @staticmethod
    def read_excel(excel_path, sheet_name: str=None) -> pd.DataFrame:
        if path.exists(excel_path):
            with open(excel_path, 'rb') as f:
                # pd.read_excel(f, sheet_name=None) returns a dict not a DataFrame
                if sheet_name is None:
                    df = pd.read_excel(f)
                else:
                    df = pd.read_excel(f, sheet_name=sheet_name)
            return df
        else:
            log.warning('File {0} doesn\'t exists.'.format(excel_path))
            return None

    def select(self, columns=None, filter=None) -> pd.DataFrame:
        """
        Selects and filters data from the self.xslx_content DataFrame.

        Parameters:
        -----------
        columns : list or None (default: None)
            A list of columns to select from the DataFrame. If None, all columns are returned.
        filter : str or None (default: None)
            A filter expression, typically in the form of a string (e.g., 'age > 30').
            If None, no filtering is applied.

        Returns:
        --------
        pd.DataFrame
            A pandas DataFrame with the selected columns and applied filter.

        Workflow:
        ---------
        1. If a filter is provided, the method applies the filter expression using the `query` function
           on `self.xslx_content` (which is a pandas DataFrame loaded from an Excel file).
           The `engine='python'` allows more complex filtering using Python syntax.
        2. If no filter is provided, the entire DataFrame is used without filtering.
        3. If specific columns are provided, only those columns are selected from the DataFrame.
        4. The final DataFrame (filtered and/or column-selected) is returned.

        Example:
        --------
        To select the 'name' and 'age' columns for rows where 'age' is greater than 30:
        >> df = self.select(columns=['name', 'age'], filter='age > 30')
        """
        if filter is not None:
            edf = self.xslx_content.query(filter, engine='python')
        else:
            edf = self.xslx_content

        if columns is not None:
            edf = edf[columns]

        return edf

    def append_excels(self, excel_paths: List[str]):
        """
        It append rows of other excels to the end of self.xslx_content.
        Columns not in the original dataframes are added as new columns and the new cells are populated with NaN value.

        :param excel_list:
        :return:
        """

        dfs = list()
        for path in excel_paths:
            dfs.append(ExcelFileHandler.read_excel(path))

        self.append_dataframe(dfs)

    def append_dataframe(self, df_list: List[pd.DataFrame]):
        """
        It appends rows of other dataframes to the end of self.xslx_content.
        Columns not in the original dataframe are added as new columns and the new cells are populated with NaN value.
        :param excel_list:
        :return:
        """

        if df_list is not None and len(df_list) > 0:
            new_xslx_content = []
            new_xslx_content.append(self.xslx_content)
            for df in df_list:
                new_xslx_content.append(df)

            self.xslx_content = pd.concat(new_xslx_content, ignore_index=True, sort=False)

    def nice_print(self, columns=None, filter=None):
        df = self.select(columns=columns, filter=filter)

        if df is not None:
            print(df.to_string())

    def write_excel(self, dest_path="output.xlsx"):
        self.xslx_content.to_excel(dest_path, index=False)

    def write_csv(self, dest_path: str=None, sep: str=None):

        if dest_path is None:
            dest_path = "output.csv"

        if sep is None:
            sep = ";"

        print(sep)
        self.xslx_content.to_csv(dest_path, sep=sep, encoding="utf-8", index=False)

