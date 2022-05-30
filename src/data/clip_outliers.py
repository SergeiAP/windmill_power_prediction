import pandas as pd
from typing import Dict
import copy

class ClipOtliers:
    feature = 'wp'

    def __init__(self, param_dict: Dict):
        self.param_dict = param_dict
        self.method = param_dict['fun']

    def get_dfs(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        dfs = copy.deepcopy(dfs)
        for df_name in dfs:
            if self.feature in df_name:
                dfs[df_name] = self.get_df(dfs[df_name])
        return dfs

    def get_df(self, df: pd.DataFrame) -> pd.DataFrame:
        param_dict = copy.deepcopy(self.param_dict)
        param_dict['cols'] = df.columns[~df.columns.isin(param_dict['exclude'])]
        df = self.method(df, param_dict)
        return df


    @staticmethod
    def quantile_clip(df: pd.DataFrame, param_dict: Dict) -> pd.DataFrame:
        """
        param_dict example:

            {'cols': df_wp1.columns[~df_wp1.columns.isin(['date'])],
                    'lower': 0.05,
                    'upper': 0.95
            }
        """
        df = df.copy()
        df[param_dict['cols']] = df[param_dict['cols']].clip(lower=df[param_dict['cols']].quantile(param_dict['lower']),
                                                            upper=df[param_dict['cols']].quantile(param_dict['upper']),
                                                            axis='columns')
        return df

    @staticmethod
    def iqr_clip(df: pd.DataFrame, param_dict: Dict) -> pd.DataFrame:
        """
        param_dict example:

            {'cols': df_wp1.columns[~df_wp1.columns.isin(['date'])]}
        """
        df = df.copy()
        q1, q3 = df[param_dict['cols']].quantile(0.25), df[param_dict['cols']].quantile(0.75)
        iqr = q3 - q1
        df[param_dict['cols']]  = df[param_dict['cols']].clip(lower=q1 - 1.5 * iqr,
                                                              upper=q3 + 1.5 * iqr,
                                                              axis='columns')
        return df
    