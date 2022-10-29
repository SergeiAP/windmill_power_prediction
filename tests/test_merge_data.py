# pylint: disable=missing-module-docstring
import great_expectations as ge
import pandas as pd
from click.testing import CliRunner
from src.data.merge_data import run_merging_data
from src.read_config import get_data_config

# Initialize runner
runner = CliRunner()

def test_cli_command() -> None:
    """Test whether the method is working by CLI generally"""
    (args, out) = get_data_config("vars", {"merge_data": ["args", "out"]},
                                  path = "./dvc.yaml",
                                  is_convert_to_dict = True)
    result = runner.invoke(run_merging_data, args + " " + out)  # type: ignore
    assert result.exit_code == 0


def test_data_output():
    """Test data output format"""
    (out,) = get_data_config("vars", {"merge_data": ["out"]},
                             path = "./dvc.yaml",
                             is_convert_to_dict = True)
    (data_types,) = get_data_config("merge_data", ["data_types"])
    df_ge = ge.from_pandas(pd.read_csv(out).astype(data_types))  # type: ignore
    
    expected_columns = ["date", "hors", "u", "v", "ws", "wd", "wp", "windfarm"]
    # Some could be null for some rows - will be splitted later
    not_null_columns = ["date", "hors", "windfarm"]
    columns_types = {"int16": ["hors"],
                     "object": ["windfarm"],
                     "float16": ["u", "v", "ws", "wd", "wp"]}
    date_format = {"columns": ["date"], "format": "%Y-%m-%d %H:%M:%S"}
    
    # General checks
    assert (
        df_ge
        .expect_table_columns_to_match_ordered_list(column_list=expected_columns)
        .success is True)
    assert (
        df_ge
        .expect_compound_columns_to_be_unique(["date", "hors", "u", "windfarm"])
        .success is True)
    # Check nulls
    df_ge["source_null_count"] = df_ge[not_null_columns].isnull().sum(axis = 1)
    assert (
        df_ge
        .expect_column_values_to_be_in_set("source_null_count", [0])
        .success is True)
    # Check types
    check_data_output_types(df_ge, columns_types, date_format)
    # Check values, wp is normalized
    assert (df_ge
            .expect_column_values_to_be_between(column="wp",
                                                min_value = 0,
                                                max_value = 1)
            .success is True)
    

def check_data_output_types(df_ge: pd.DataFrame,
                            columns_types: dict[str, list[str]],
                            date_fomrat: dict[str, list | str]) -> None:
    """Check data types for specicif DataFrame

    Args:
        df_ge (pd.DataFrame): Dataframe to be checked
        columns_types (dict[str, list[str]]): types and corresponding columns to be 
        checked
        date_fomrat (dict[str, list | str]): date columns and their format to be 
        checked
    """
    for coltype in columns_types:
        for col in columns_types[coltype]:
            assert (
                df_ge
                .expect_column_values_to_be_of_type(column=col, type_=coltype)
                .success is True)
    for datecol in date_fomrat["columns"]:
        assert (
            df_ge
            .expect_column_values_to_match_strftime_format(
                column=datecol, strftime_format=date_fomrat["format"])
            .success is True)
