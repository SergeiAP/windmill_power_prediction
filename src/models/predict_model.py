# pylint: disable=missing-module-docstring
# TODO: replace print to log
import pickle  # nosec B403
import time

import click
import pandas as pd
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
    (date_col, target_col) = get_data_config("common", ["date_col", "target"])
    df = pd.read_csv(input_filepath)
    with open(input_modelpath, 'rb') as f:
        model = pickle.load(f)

    dates = df[date_col]
    df = (df.loc[:, features["include"]] if isinstance(features["include"], list)
          else df)
    df = df.drop(features["exclude"], axis="columns")
    
    predictions = model.predict(df)
    df_pred = pd.concat(
        [dates, pd.Series(predictions, name=target_col)], axis="columns")
    start_date, end_date = df_pred[date_col].min(), df_pred[date_col].max()
    print(f"Predicted {len(predictions)} rows from {start_date} to {end_date}")
    
    print(f"Execution time is {round((time.time() - start_time) / 60, 2)} minutes")
    df_pred.to_csv(output_filepath, index=False)

if __name__ == "__main__":
    predict()  # pylint: disable=no-value-for-parameter
