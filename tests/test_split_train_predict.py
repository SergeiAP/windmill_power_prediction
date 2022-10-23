# pylint: disable=missing-module-docstring
import great_expectations as ge
import pandas as pd
from click.testing import CliRunner
from src.data.split_train_predict import split_train_predict
from src.read_config import get_data_config

# Initialize runner
runner = CliRunner()

def test_cli_command() -> None:
    """Test whether the method is working by CLI generally"""
    (args, out) = get_data_config("vars", {"split_train_predict": ["args", "out"]},
                                  path = "./dvc.yaml",
                                  is_convert_to_dict = True)
    result = runner.invoke(split_train_predict, args + " " + out) # type: ignore
    assert result.exit_code == 0


def test_data_output():
    """Test data output format"""
    (out,) = get_data_config("vars", {"split_train_predict": ["out"]},
                             path = "./dvc.yaml",
                             is_convert_to_dict = True)
    df_ge = ge.from_pandas(pd.read_csv(out))
    
    df_ge["source_null_count"] = df_ge.isnull().sum(axis = 1)
    assert df_ge.expect_column_values_to_be_in_set("source_null_count", [0]).success is True  # pylint: disable=line-too-long
    