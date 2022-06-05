# pylint: disable=missing-module-docstring
# TODO: make print as log
import copy
import time
from collections import OrderedDict

import click
import pandas as pd

# pylint: disable=import-error
from create_features_config import feature_dict, return_feature_order

# ================================ Feature aggregator class ==================

class FeatureCreator:
    """
    Main class to create and accumulate all features
    """
    # Set column to iterate among windfarms
    windfarm_col = 'windfarm'

    def __init__(self,
                 general_params: dict,
                 specific_params: dict,
                 feature_order: OrderedDict
                 ) -> None:
        """Set params for feature creation

        Args:
            general_params (dict): common params useful for each feature 
            specific_params (dict): each feature parameters
            feature_order (OrderedDict): process order of features
        """
        self.general_params = general_params
        self.attach_params: dict | None = (
            specific_params.pop('attach') if specific_params.get('attach')
            else None
        )
        self.specific_params = specific_params
        self.feature_order = feature_order
        self.del_rows: int = 0

    def get_dfs(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Aimed to process df with all windfarms to create all additional features

        Args:
            df (pd.DataFrame): dataframe with primary features and `windfarm_col`
            column

        Returns:
            tuple[pd.DataFrame, int]: processed dataframe and deleted rows
        """
        dfs_list: list[pd.DataFrame] = []
        df = copy.deepcopy(df)
        del_rows = 0
        if self.attach_params and self.attach_params.get('attach'):
            df = AttachFeature.attach(
                path=self.attach_params['path'],
                df_dataset=df,
                feature_name=self.attach_params['feature_name'],
                predicted_dates=self.attach_params['predicted_dates']
        )
        for windfarm_id in df[self.windfarm_col].unique():
            df_farm = df.loc[df[self.windfarm_col] == windfarm_id]
            df_farm, del_rows = self.get_df(df_farm)
            dfs_list.append(df_farm)
        df_res = pd.concat(dfs_list, axis="index") 
        return df_res, del_rows
 
    def get_df(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Aimed to process one wp DataFrames

        Args:
            df (pd.DataFrame): dataframe with data for one windfarm

        Returns:
            tuple[pd.DataFrame, int]: processed dataframe and deleted rows
        """
        for key, func in self.feature_order.items():
            df, self.del_rows = func.create(df=df,
                                            del_rows=self.del_rows,
                                            general_params=self.general_params,
                                            specific_params=self.specific_params[key])
        adds_del = (
            self.general_params.get('additional_drops') 
            if self.general_params.get('additional_drops')
            else 0
        )
        del_rows = self.del_rows + adds_del # type: ignore
        df = self.cut_head(df=df,
                           del_rows=del_rows,
                           reset_index=True)
        return df, del_rows

    @staticmethod
    def cut_head(df: pd.DataFrame,
                 del_rows: int,
                 reset_index: bool = True
                 ) -> pd.DataFrame:
        """Delete specific number of rows in the head

        Args:
            df (pd.DataFrame): dataframe to be cut
            del_rows (int): number of rows to be deleted
            reset_index (bool, optional): whether to reset index. Defaults to True.

        Returns:
            pd.DataFrame: cut dataframe
        """
        print("Rows to del: ", del_rows)
        df.drop(df.head(del_rows).index, inplace=True)
        if reset_index:
            df.reset_index(drop=True, inplace=True)
        return df


class AttachFeature:
    # TODO: required to chek after refactoring
    """
    Get earlier predicted wp features to use it in new prediction
    """
    date_col = 'date'
    targte_name = 'wp'
    windfarm_col = 'windfarm'

    @classmethod
    def attach(cls,
               path: str,
               df_dataset: pd.DataFrame,
               feature_name: str = 'wp_pred',
               predicted_dates: list[str] | None = None
               ) -> pd.DataFrame:
        """Attach traget (`target_name`) feature for dataset

        Args:
            path (str): path for `target_name` feature with new values
            df_dataset (pd.DataFrame): dataframe to attach|replace|add `target_name`
            feature_name (str, optional): new name. Defaults to 'wp_pred'.
            predicted_dates (list[str] | None, optional): if required set dates for 
            replacing values by `path` dataset. Defaults to None.

        Returns:
            pd.DataFrame: dataframe with `feature_name` column
        """
        df_dataset = copy.deepcopy(df_dataset)
        df_predicted = pd.read_csv(path)
        df_predicted[cls.date_col] = pd.to_datetime(df_predicted[cls.date_col])
        if predicted_dates:
            df_predicted = cls.filter_dates(df_predicted,
                                            cls.date_col,
                                            predicted_dates)
        df_predicted = cls.transform_data(df_predicted,
                                          cls.targte_name,
                                          cls.windfarm_col)
        df_merged = cls.merge_dfs(df_predicted,
                                  df_dataset,
                                  cls.date_col,
                                  cls.targte_name,
                                  cls.windfarm_col,
                                  feature_name)
        
        cls.check_dates(df_merged, df_dataset, cls.date_col)
        return df_merged

    @staticmethod
    def filter_dates(df_predicted: pd.DataFrame,
                     date_col: str,
                     predicted_dates: list[str]
                     ) -> pd.DataFrame:
        """Filter dataset by predicted_dates 

        Args:
            df_predicted (pd.DataFrame): dataframe to be filtered
            date_col (str): date column
            predicted_dates (list[str]): data to be included

        Returns:
            pd.DataFrame: _description_
        """
        return df_predicted[df_predicted[date_col].isin(predicted_dates)]


    @staticmethod
    def transform_data(df_predicted: pd.DataFrame,
                       targte_name: str,
                       windfarm_col: str
                       ) -> pd.DataFrame:
        """Change data with predicted vals for common train dataset schema

        Args:
            df_predicted (pd.DataFrame): predicted vals
            targte_name (str): target column name
            windfarm_col (str): column denoted windfarm

        Returns:
            pd.DataFrame: transformed dataset
        """
        dfs_list: list[pd.DataFrame] = []
        for col in df_predicted.columns:
            df_windfarm = df_predicted[[col]].rename({col: targte_name})
            df_windfarm[windfarm_col] = col
            dfs_list.append(df_windfarm)
        return pd.concat(dfs_list, axis="index")

    @staticmethod
    def merge_dfs(df_predicted: pd.DataFrame,
                  df_train: pd.DataFrame,
                  date_col: str,
                  targte_name: str,
                  windfarm_col: str,
                  feature_name: str
                  ) -> pd.DataFrame:
        """Merge train and predicted datasets

        Args:
            df_predicted (pd.DataFrame): dataframe with predicted values
            df_train (pd.DataFrame): dataframe with primary features
            date_col (str): date column name
            targte_name (str): target column name
            windfarm_col (str): column denoted windfarm
            feature_name (str): new feature name with predicted values

        Returns:
            pd.DataFrame: _description_
        """
        dfs_list: list[pd.DataFrame] = []
        for windfarm_id in df_train[windfarm_col].unique():
            df_farm = df_train.loc[df_train[windfarm_col] == windfarm_id]
            df_farm[feature_name] = df_farm[targte_name].copy()
            
            date_filter = df_farm[date_col].isin(df_predicted[date_col])
            nan_target_filter = df_farm[feature_name].isna()
            common_dates = df_farm[date_col].loc[date_filter & nan_target_filter]
            
            df_farm.loc[df_farm[date_col].isin(common_dates), feature_name] = (
                df_predicted.loc[df_predicted[date_col].isin(common_dates)])
        return pd.concat(dfs_list, axis="index")

    @staticmethod
    def check_dates(df_check: pd.DataFrame,
                    df_reference: pd.DataFrame,
                    date_col: str) -> None:
        """Check two datasets to have almost the same dates

        Args:
            df_check (pd.DataFrame): first dataset
            df_reference (pd.DataFrame): second dataset
            date_col (str): date column name
        """
        assert (df_check[date_col].to_numpy() 
                == df_reference[date_col].to_numpy()).all()


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def run_create_features(input_filepath: str, output_filepath: str) -> None:
    """Create new features in clipped dataset

    Args:
        input_filepath (str): path to the clipped dataset
        output_filepath (str): path to save dataset with new features
    """
    start_time = time.time()
    
    df = pd.read_csv(input_filepath)
    
    feature_order = return_feature_order()
    df, del_rows = FeatureCreator(general_params=feature_dict['general_params'],
                                  specific_params=feature_dict,
                                  feature_order=feature_order).get_dfs(df=df)
    df.to_csv(output_filepath, index=False)
    
    print(f"Deleted rows is: {del_rows}")
    print(f"Execution time is {round((time.time() - start_time) / 60, 2)} minutes")


if __name__ == "__main__":
    run_create_features()  # pylint: disable=no-value-for-parameter
