# pylint: disable=missing-module-docstring
# to import modules from src
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from src.read_config import get_data_config


class WeatherPredictionHandler(ABC):
    """Interface for other weather aggregation strategies"""

    @classmethod
    @abstractmethod
    def get_df(cls, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        Launch handler which process several weather predictions to use dataset in
        further ML steps

        Args:
            df (pd.DataFrame): dataset with not prepared wether predictions
            params (dict): parameters for dedicated transformation

        Returns:
            pd.DataFrame: transformed dataset with aggregated weather parameters
        """


class WideWeatherPrediction(WeatherPredictionHandler):
    # TODO: required to chek after refactoring
    """
    Transform a dataset into 4 weather predictions.
    Each time stamp has 1 to 4 forecasts.

    Input params example:
        {'date_col': 'date',
        'cols_to_transform': ['u', 'v', 'ws', 'wd'],
        'reset_index': True}
    """

    @classmethod
    def get_df(cls, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        df = df.copy()
        dfs_list: list[pd.DataFrame] = []
        date_col = params["date_col"]
        for col in params["cols_to_transform"]:
            df_pivoted = df.pivot_table(
                index=date_col,
                columns=(df.groupby(date_col).cumcount()),  # type: ignore
                values=col,
                aggfunc="sum",
            ).add_prefix(col + "_")
            dfs_list.append(df_pivoted)
        df_res = pd.concat(dfs_list, axis=1)
        return df_res.reset_index() if params["reset_index"] else df_res


class OrderWeatherPrediction(WeatherPredictionHandler):
    """
    Give a new df from df grouped by date with a certain hours order (from
    freshness hour to the oldest one) for each date group. Extended class of
    FreshWeatherPrediction (with descending_order=3 == Fresh class).
    Class could be used to create 4 train datasets from 4 predictions.

    Extended class of FreshWeatherPrediction

    Input params example:
        {'sort': ['date', 'hors'],
        'date_col': 'date',
        'grouby_col': 'date',
        'col_idx': 'hors',
        'col_na_check': 'ws',
        'descending_order': -1,}  # Optional: [0, 1, 2, 3] or -1 - random
    """

    (seed,) = get_data_config("common", ["seed"])

    @classmethod
    def get_df(cls, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        df = df.copy()
        df = df[df[params["col_na_check"]].notna()].reset_index(drop=True)
        if params["descending_order"] != -1:
            df = df.groupby(params["grouby_col"], as_index=False).apply(
                lambda x: x.sort_values(by=params["col_idx"], ascending=False).iloc[
                    params["descending_order"]
                ]
            )
        else:
            df = df.groupby(params["grouby_col"], as_index=False).apply(
                lambda x: x.sample(1, random_state=1)
            )
        df.sort_values(params["sort"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df


class FreshWeatherPrediction(WeatherPredictionHandler):
    """
    Give a new df with fresh-only predictions

    Input params example:
        {'sort': ['date', 'hors'],
        'grouby_col': 'date',
        'col_idx_min': 'hors',
        'col_na_check': 'ws',
        'check': True}  # Works for wp1 only
    """

    @classmethod
    def get_df(cls, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        df = df.copy()
        df = df[df[params["col_na_check"]].notna()].reset_index(drop=True)
        df = df.loc[df.groupby(params["grouby_col"])[params["col_idx_min"]].idxmin()]
        df.sort_values(params["sort"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        if params["check"]:
            cls.check(df=df)
        return df

    @staticmethod
    def check(df: pd.DataFrame) -> None:
        # TODO: A bit hardcoded:-)
        # TODO: works for wp1 only
        # NOTE: function to check:
        mean_dict = {
            "2009-08-01 01:00:00": 1.88,
            # First test date for second period + '2011-01-04 13:00:00'
            # the same
            "2012-06-19 13:00:00": 5.08,
            "2012-06-20 01:00:00": 6.97,
            # 13th test date for second period (3 prediction periods available)
            # + '2011-01-05 01:00:00' the same
            "2012-06-21 01:00:00": 8.230,
            # 37th test date for second period (1 prediction period available)
            # + '2011-01-06 01:00:00' the same
            "2012-06-25 00:00:00": 4.910,
        }  # Last date for test
        for key in mean_dict:
            diff = abs(float(df[df.date == key]["ws"]) - mean_dict[key])
            if not diff < 0.01:
                raise AssertionError(
                    f"Values are different for ws column for date {key}, diff={diff}"
                )
        print("Status: SUCCESS")


class AggregatedWeatherPrediction(WeatherPredictionHandler):
    # TODO: required to chek after refactoring
    # TODO: class a bit forgotten because others better for the task
    """
    Give a new df with aggregated predictions

    Input params example:
        {'date_col': 'date',
        'grouped_length': 4*(36+48),
        'sort': ['date', 'hors'],
        'extend_df': [{'date': '2012-06-23 13:00:00',
                    'hors': 36},
                    {'date': '2012-06-24 01:00:00',
                    'hors': 24},
                    {'date': '2012-06-24 13:00:00',
                    'hors': 12},],
        'fit_to_test_params': {'mul': 4,
                            'group_len': 36+48,
                            'group_start_idx': 36+12,
                            'group_start_hors': [1, 13, 25],
                            'group_end_hors': [12, 24, 36]},
        'agg_params': {'wd_sin_cos_transform': True,
                    'wd_col': 'wd',
                    'wd_drop': True,
                    'exclude_cols': ['hors'],
                    'agg_fun': 'mean'},
        'check': True}  # Works for wp1 only
    """

    @classmethod
    def get_df(cls, df: pd.DataFrame, params: dict):
        df = df.copy()
        df = cls.extend_df(
            df=df, date_and_hors=params["extend_df"], date_col=params["date_col"]
        )
        divider = (
            params["fit_to_test_params"]["mul"]
            * params["fit_to_test_params"]["group_len"]
        )
        df.sort_values(params["sort"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        cls.check_len_integer(length=len(df), divider=divider)
        df = cls.fit_to_test_df_structure(df, params["fit_to_test_params"])
        df = cls.aggregate_weather_predictions(
            df=df, date_col=params["date_col"], agg_params=params["agg_params"]
        )
        if params["check"]:
            cls.check(df=df, method=params["agg_params"]["agg_fun"])
        return df

    @staticmethod
    def check_len_integer(length: int, divider: int):
        assert (length / divider).is_integer()

    @staticmethod
    def extend_df(
        df: pd.DataFrame, date_and_hors: list[dict], date_col: str
    ) -> pd.DataFrame:
        """
        Extend df based on start_date and hors in date_and_hors without end_date
        to have the same df structure for the whole df
        """
        df_copy = df.copy()
        for elem in date_and_hors:
            hors = elem["hors"]
            date_filter = df_copy.date >= elem["date"]
            extend_dict = {
                "date": df_copy[date_filter][date_col].unique(),
                "hors": list(range(1, hors + 1)),
                "u": [0] * hors,
                "v": [0] * hors,
                "ws": [0] * hors,
                "wd": [0] * hors,
            }
            df_copy = pd.concat([df_copy, pd.DataFrame.from_dict(extend_dict)])
        return df_copy

    @staticmethod
    def fit_to_test_df_structure(
        df: pd.DataFrame, fit_to_test_params: dict
    ) -> pd.DataFrame:
        df_copy = df.copy()
        mul = fit_to_test_params["mul"]
        grouped_length = mul * fit_to_test_params["group_len"]

        for grouped_idx in range(0, int(len(df_copy) / grouped_length)):
            start_idx = (
                grouped_idx * grouped_length
                + mul * fit_to_test_params["group_start_idx"]
            )
            end_idx = (grouped_idx + 1) * grouped_length
            start_hors = [1, 13, 25]
            end_hors = [12, 24, 36]
            start_idxs = [start_idx]

            for idx in range(len(start_hors[1:])):
                start_idxs.append(
                    start_idx + mul * (end_hors[idx] - start_hors[idx] + 1)
                )
            elems_for_start_end_filter = zip(
                fit_to_test_params["group_start_hors"],
                fit_to_test_params["group_end_hors"],
                start_idxs,
            )
            for start_hor, end_hor, inter_idx in elems_for_start_end_filter:
                df_copy_part = df_copy.loc[inter_idx:end_idx, :]
                start_end_filter = (start_hor <= df_copy_part["hors"]) & (
                    df_copy_part["hors"] <= end_hor
                )
                df_copy.drop(df_copy_part[start_end_filter].index, inplace=True)
        return df_copy

    @staticmethod
    def aggregate_weather_predictions(
        df: pd.DataFrame, date_col: str, agg_params: dict
    ) -> pd.DataFrame:
        df_copy = df.copy()
        if agg_params["wd_sin_cos_transform"]:
            df_copy["wd_sin"] = np.sin(np.deg2rad(df_copy[agg_params["wd_col"]]))
            df_copy["wd_cos"] = np.cos(np.deg2rad(df_copy[agg_params["wd_col"]]))
            if agg_params["wd_drop"]:
                df_copy.drop(columns=[agg_params["wd_col"]], inplace=True)
        cols_to_agg = df_copy.columns[~df_copy.columns.isin(agg_params["exclude_cols"])]
        df_copy = (
            df_copy[cols_to_agg]  # type: ignore
            .groupby(date_col, as_index=False)
            .agg(agg_params["agg_fun"])
        )
        return df_copy

    @staticmethod
    def check(df: pd.DataFrame, method: str) -> None:
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            method (str): _description_

        Raises:
            AssertionError: _description_
        """
        # TODO: A bit hardcoded:-)
        # NOTE: function to check:
        if method == "mean":
            mean_dict = {
                "2009-08-01 01:00:00": 1.7425,
                # First test date for second period + '2011-01-04 13:00:00'
                # the same
                "2012-06-19 13:00:00": 6.005,
                # 13th test date for second period (3 prediction periods
                # available) + '2011-01-05 01:00:00' the same
                "2012-06-20 01:00:00": 6.2667,
                # 37th test date for second period (1 prediction period
                # available) + '2011-01-06 01:00:00' the same
                "2012-06-21 01:00:00": 8.230,
                "2012-06-25 00:00:00": 4.910,
            }  # Last date for test
            for key, values in mean_dict.items():
                diff = abs(float(df[df.date == key]["ws"]) - values)
                if not diff < 0.01:
                    raise AssertionError(
                        f"Values are different for ws column for date"
                        f"{key}, diff={diff}"
                    )
            print("Status: SUCCESS")
        else:
            print(f"Method {method} not exists for chechking")
