# pylint: disable=missing-module-docstring
# pylint: disable=wrong-import-position
import sys

ROOT_FOLDER = "./"
sys.path.append(ROOT_FOLDER)

import copy

import click
import pandas as pd
from src.read_config import get_data_config


CLIP_STRATEGY = 'quantile_clip'


class ClipOtliers:
    """Clip data for specific columns to reduce outliers"""
    date_col = 'date'
    windfarm_col = 'windfarm'
    target_col = 'wp'
    

    def __init__(self, param_dict: dict) -> None:
        """Initialize class

        Args:
            param_dict (dict): dict with clip method and exclusions
        """
        self.clip_methods = {'quantile_clip': self.quantile_clip,
                             'iqr_clip': self.iqr_clip}
        self.param_dict = param_dict
        self.method = self.clip_methods[param_dict['fun']]

    def get_dfs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip dataset for each windfarm in column `windfarm_col` 

        Args:
            df (pd.DataFrame): dataframe with column `windfarm_col` to clip columns

        Returns:
            pd.DataFrame: cliped dataframe for each windfarm
        """
        dfs_list: list[pd.DataFrame] = []
        for windfarm_id in df[self.windfarm_col].unique():
            df_farm = df.loc[df[self.windfarm_col] == windfarm_id]
            dfs_list.append(self.get_df(df_farm))
        df_res = pd.concat(dfs_list, axis="index")
        return df_res

    def get_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """clip dataset of one windfarm 

        Args:
            df (pd.DataFrame): one windfarm dataset

        Returns:
            pd.DataFrame: cliped windfarm ignoring `exclude` cols in `self.param_dict`
        """
        param_dict = copy.deepcopy(self.param_dict)
        param_dict['cols'] = df.columns[~df.columns.isin(param_dict['exclude'])]
        df = self.method(df, param_dict)
        return df


    @staticmethod
    def quantile_clip(df: pd.DataFrame, param_dict: dict) -> pd.DataFrame:
        """Clip data by quantile values

        Args:
            df (pd.DataFrame): dataframe of one windfarm to be cliped
            param_dict (dict): params for cliping and exclusions of it
            param_dict example:
                {'cols': df_wp1.columns[~df_wp1.columns.isin(['date'])],
                'lower': 0.05,
                'upper': 0.95}

        Returns:
            pd.DataFrame: cliped dataset
        """
        df = df.copy()
        lower_q = df[param_dict['cols']].quantile(param_dict['lower'])
        upper_q = df[param_dict['cols']].quantile(param_dict['upper'])
        df[param_dict['cols']] = df[param_dict['cols']].clip(
            lower=lower_q, upper=upper_q, axis='columns' # type: ignore
        )
        return df

    @staticmethod
    def iqr_clip(df: pd.DataFrame, param_dict: dict) -> pd.DataFrame:
        """Clip data by IQR values

        Args:
            df (pd.DataFrame): dataframe of one windfarm to be cliped
            param_dict (dict): params for cliping and exclusions of it
            param_dict example:
                {'cols': df_wp1.columns[~df_wp1.columns.isin(['date'])]}

        Returns:
            pd.DataFrame: cliped dataset
        """
        df = df.copy()
        q1 = df[param_dict['cols']].quantile(0.25)
        q3 = df[param_dict['cols']].quantile(0.75)
        iqr = q3 - q1
        df[param_dict['cols']]  = df[param_dict['cols']].clip(
            lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr, axis='columns' # type: ignore
        )
        return df


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def run_clip_outliers(input_filepath: str, output_filepath: str) -> None:
    """Clip all columns excpet excluded in `clip_outliers` for dataset with weveral
    windfarms

    Args:
        input_filepath (str): dataset with several windfarms in column `windfarm_col`
        output_filepath (str): path to save dataset with clipped dataset
    """
    # read section
    (clip_params,) = get_data_config('clip_outliers', [CLIP_STRATEGY])
    clip_params.update({'fun': CLIP_STRATEGY})
    df = pd.read_csv(input_filepath)
    
    df = ClipOtliers(clip_params).get_dfs(df=df)
    df.to_csv(output_filepath, index=False)


if __name__ == "__main__":
    run_clip_outliers()  # pylint: disable=no-value-for-parameter
