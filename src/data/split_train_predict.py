# pylint: disable=missing-module-docstring
# TODO: change print to logs
import click
import pandas as pd

from src.read_config import get_data_config


def filter_df_by_date(df: pd.DataFrame,
                      date_col: str,
                      target_col: str,
                      params: dict) -> pd.DataFrame:
    date_filter = ((df[date_col] >= params['dates'][0])
                   & (df[date_col] <= params['dates'][1]))
    filter_ = ((df[target_col].isna() & date_filter) if params["is_na"]
               else (df[target_col].notna() & date_filter))
    return df.loc[filter_, :]


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_trainpath", type=click.Path())
@click.argument("output_predictpath", type=click.Path())
def split_train_predict(input_filepath: str,
                        output_trainpath: str,
                        output_predictpath: str) -> None:
    # read section
    df = pd.read_csv(input_filepath)
    target_col, date_col = get_data_config("common",
                                           ["target", "date_col"])
    train_rows, predict_rows = get_data_config("split_train_predict",
                                               ["train_rows", "predict_rows"])
    
    df_train = filter_df_by_date(df, date_col, target_col, train_rows)
    df_predict = filter_df_by_date(df, date_col, target_col, predict_rows)
    
    df_train.to_csv(output_trainpath, index=False)
    df_predict.to_csv(output_predictpath, index=False)


if __name__ == "__main__":
    split_train_predict()  # pylint: disable=no-value-for-parameter
