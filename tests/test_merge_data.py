from click.testing import CliRunner
from src.data.merge_data import run_merging_data
from src.read_config import get_data_config


# Initialize runner
runner = CliRunner()


def test_cli_command() -> None:
    """Test whether the method is working by CLI generally"""
    (args, ) = get_data_config("vars", {"merge_data": ["args"]},
                               path = "./dvc.yaml",
                               is_convert_to_dict = True)
    result = runner.invoke(run_merging_data, args) # type: ignore
    assert result.exit_code == 0
    
