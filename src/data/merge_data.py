# pylint: disable=missing-module-docstring
from pathlib import Path
from typing import Optional

import click
import pandas as pd

DATA_PATHS = {
    "wp1": "./wp1.csv",
    "wp2": "./wp2.csv",
    "wp3": "./wp3.csv",
    "wp4": "./wp4.csv",
    "wp5": "./wp5.csv",
    "wp6": "./wp6.csv",
    "train": "./train.csv",
    "test": "./test.csv",
}
# Last date for test dataset
LAST_TEST_DATE = "2012-06-25 00:00:00"
# First date for train dataset with all 4 weather predictions
FIRST_TRAIN_DATE = "2009-07-02 13:00:00"


class DataMerger:
    """
    Class aimed to connect to .csv's and make merge of it
    df_datasets = DataConnection(
                    target_name=TARGET_NAME,
                    filter_date=(FIRST_TRAIN_DATE, LAST_TEST_DATE)
                    ).get_df_dict(DATA_PATHS)
    """

    date_colname = "date"
    feature = "wp"
    target_name = "wp"
    windfarm_col = "windfarm"
    params = {
        "test": {"hors_col": None, "sort_cols": [date_colname]},
        "train": {"hors_col": None, "sort_cols": [date_colname]},
        feature: {"hors_col": "hors", "sort_cols": [date_colname]},
    }

    def __init__(self,
                 target_name: str = "wp",
                 filter_date: tuple = ("2009-07-02 13:00:00", "2012-06-25 00:00:00"),
                 ) -> None:
        self.params["train"].update(
            {
                "filter": {
                    "col": self.date_colname,
                    "lower": filter_date[0],
                    "upper": filter_date[1],
                }
            }
        )
        self.params[self.feature].update(
            {
                "filter": {
                    "col": self.date_colname,
                    "lower": filter_date[0],
                    "upper": filter_date[1],
                }
            }
        )
        self.target_name = target_name

    def merge_data(self, paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return test and merge train datasets into one long table format"""
        df_dict = self.get_df_dict(paths)
        df_dict = self.integrate_target(df_dict)
        df_test, df_train = self.merge_df_features(df_dict)
        return (df_test, df_train)

    def get_df_dict(self, paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
        """Read dataset, convert data and pack into dict"""
        df_dict = {}
        for df_name in paths:
            selected_params = self.get_params(df_name)
            df_dict[df_name] = self.get_features(paths[df_name], selected_params)
        return df_dict

    def integrate_target(
        self, df_dict: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Put target into fetures datasets and then delete initial dataset"""
        for col in df_dict["train"].columns:
            df_dict[col][self.target_name] = df_dict["train"][col]
        del df_dict["train"]
        return df_dict

    def merge_df_features(
        self, df_dict: dict[str, pd.DataFrame]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Merge features into one train dtaset"""
        df_merged_list = []
        dict_keys = [df_name for df_name in df_dict if self.feature in df_name]
        for df_name in dict_keys:
            df_dict[df_name][self.windfarm_col] = df_name
            df_merged_list.append(df_dict[df_name])
        return (
            df_dict["test"],
            pd.concat(df_merged_list, axis="index"),
        )  # type: ignore

    def get_features(self, path: Path, params: dict) -> pd.DataFrame:
        """Convert date, sort and filter data"""
        df = pd.read_csv(path)
        df = self.set_date_format(df=df, hors_col=params["hors_col"])
        df.set_index(self.date_colname, drop=True, inplace=True)
        df.sort_index(inplace=True)
        if "filter" in params:
            df = self.filter_date(
                df=df,
                lower=params["filter"]["lower"],
                upper=params["filter"]["upper"],
            )
        return df

    def get_params(self, df_name: str):
        """Get params for specific files"""
        if self.feature in df_name:
            return self.params[self.feature]
        return self.params[df_name]

    @staticmethod
    def set_date_format(df: pd.DataFrame,
                        hors_col: Optional[str] = "hors",
                        newcol: Optional[str] = None,
                        date_format: str = "%Y%m%d%H",
                        ) -> pd.DataFrame:
        """Set date format and convert column into date format"""
        df.date = pd.to_datetime(df.date, format=date_format)

        if isinstance(hors_col, str):
            date_col = df.date + df[hors_col].astype("timedelta64[h]")  # type: ignore
        else:
            date_col = df.date

        if isinstance(newcol, str):
            df[newcol] = date_col
        else:
            df.date = date_col
        return df

    @staticmethod
    def filter_date(df: pd.DataFrame,
                    upper: str = "2100-01-01 01:00:00",
                    lower: str = "2000-01-01 01:00:00",
                    ) -> pd.DataFrame:
        """Filter data by date range"""
        return df[(df.index >= lower) & (df.index <= upper)]  # type: ignore


@click.command()
@click.argument("input_folderpath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path(), nargs=2)
def run_merging_data(input_folderpath: str, output_filepath: str) -> None:
    """_summary_

    Args:
        input_filepath (str): _description_
        output_filepath (str): _description_
    """
    
    data_paths = {
        filename: Path(Path(".") / input_folderpath) / path 
        for filename, path in DATA_PATHS.items() # type: ignore
    }  # type: ignore
    df_test, df_train = DataMerger(
        filter_date=(FIRST_TRAIN_DATE, LAST_TEST_DATE)
    ).merge_data(data_paths)
    df_test.to_csv(output_filepath[0])
    df_train.to_csv(output_filepath[1])


if __name__ == "__main__":
    run_merging_data()  # pylint: disable=no-value-for-parameter
