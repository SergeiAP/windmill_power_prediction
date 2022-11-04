# pylint: disable=missing-module-docstring
# TODO: replace print to log
import pickle  # nosec B403
import time

import click
import pandas as pd
# For pickle
# TODO: modify as in https://stackoverflow.com/questions/54012769/saving-an-sklearn-functiontransformer-with-the-function-it-wraps
from src.models.explore_train_model import (  # pylint: disable=unused-import
    func, inverse_func)
from src.read_config import get_data_config


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("input_modelpath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def predict(input_filepath: str, input_modelpath: str, output_filepath: str) -> None:
    """Predict values using prefitted model

    Args:
        input_filepath (str): path to dataset with features for prediction
        input_modelpath (str): path to prefitted model
        output_filepath (str): path to save predictions
    """
    start_time = time.time()
    
    # read section
    (features, ) = get_data_config("explore_train_model", ["features"])
    (date_col, target_col, windfarm_col) = get_data_config(
        "common", ["date_col", "target", "windfarm_col"])
    df = pd.read_csv(input_filepath)
    with open(input_modelpath, 'rb') as file_:
        model = pickle.load(file_)

    df_pred = df[[date_col, windfarm_col]]
    df = (df.loc[:, features["include"]] if isinstance(features["include"], list)
          else df)
    df = df.drop(features["exclude"], axis="columns")
    
    predictions = pd.Series(model.predict(df), name=target_col).clip(0, 1) 
    df_pred = pd.concat([df_pred, predictions], axis="columns")
    start_date, end_date = df_pred[date_col].min(), df_pred[date_col].max()
    print(f"Predicted {len(predictions)} rows from {start_date} to {end_date}")
    
    print(f"Execution time is {round((time.time() - start_time) / 60, 2)} minutes")
    df_pred.to_csv(output_filepath, index=False)

if __name__ == "__main__":
    predict()  # pylint: disable=no-value-for-parameter
