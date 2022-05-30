# pylint: disable=missing-module-docstring
import copy

import click
import pandas as pd

# pylint: disable=import-error
from aggregate_weather_config import agg_params_train_test # agg_params_all


class WeatherAggregationPolicy:
    """
    Aggregate data by weather predictions
    Input dictionary example:
        For different strategy for train and test:
            {'train': {'class': OrderWeatherPrediction.get_df,
                       'params': order_weather_params},
            'test': {'class': FreshWeatherPrediction.get_df,
                     'params': fresh_weather_params}}
        For one strategy:
            {'all': {'class': AggregatedWeatherPrediction.get_df,
                     'params': agg_weather_params}}

    """
    date = 'date'
    windfarm = 'windfarm'
    target = 'wp'

    def __init__(self) -> None:
        self.get_df = self.get_whole_df
        
    def get_dfs(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        Interate over series of windfarms to aggregate weather features for each one to
        have unique date for each windfarm

        Args:
            df (pd.DataFrame): dataset with weather features and `windfarm` col
            params (dict): parameters to denote preprocess characteristics

        Returns:
            pd.DataFrame: dataset with unique date for windfarm and aggregated weather
        """
        dfs_list = []
        df = df.reset_index()
        params = copy.deepcopy(params)
        if 'nans' in params and 'train' in params:
            self.get_df = self.get_concat_df
            for colname in params:
                date_idxs = (
                    df[df[self.target].notna()][self.date] if colname == "train" else
                    df[df[self.target].isna()][self.date]
                )
                params[colname].update({self.date: date_idxs.sort_values()})
        for windfarm in df[self.windfarm].unique():
            df_farm = df[df[self.windfarm] == windfarm]
            dfs_list.append(self.get_df(df=df_farm, params=params)) # type: ignore
        df_res = pd.concat(dfs_list, axis="index")
        return df_res # type: ignore
    

    def get_whole_df(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            params (dict): _description_

        Raises:
            IndexError: _description_

        Returns:
            pd.DataFrame: _description_
        """
        if len(params) != 1:
            raise IndexError(f'len(params)={len(params)} should be = 1')
        for df_type in params:
            df = params[df_type]['class'](df=df, params=params[df_type]['params'])
        return df

    def get_concat_df(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Concatinate different datasets (e.g. train and test, train and missing wp's)

        Args:
            df (pd.DataFrame): dataframe with one windfarm and weather params
            params (dict): params for prerocessing

        Returns:
            pd.DataFrame: dataframe of one farm with processed weather features
        """
        dfs_list = []
        for df_type in params:
            df_agg = params[df_type]['class'](
                df=df[df[self.date].isin(params[df_type][self.date])],
                params=params[df_type]['params']
                )
            dfs_list.append(df_agg.set_index(keys=self.date, drop=True))
        return pd.concat(dfs_list).reset_index(drop=False).sort_values(self.date)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def run_aggregate_weather(input_filepath: str, output_filepath: str) -> None:
    """Aggregate weather fetures depending on imported aggregate_wetaher_config.
    By practise the best one is to take OrderWeatherPrediction for known wp and 
    FreshWeatherPrediction for unknow wp. Other strategies were tested earlier and
    couls be used also - e.g. AggregatedWeatherPrediction.

    Args:
        input_filepath (str): dataset with several windfarms in column `windfarm`
        output_filepath (str): path to save dataset with aggregated weather
    """
    df = pd.read_csv(input_filepath)
    df = WeatherAggregationPolicy().get_dfs(df=df, params=agg_params_train_test)
    df.to_csv(output_filepath, index=False)


if __name__ == "__main__":
    run_aggregate_weather()  # pylint: disable=no-value-for-parameter
