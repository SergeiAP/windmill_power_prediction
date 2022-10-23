# pylint: disable=missing-module-docstring
from click.testing import CliRunner
from src.models.predict import predict
from src.read_config import get_data_config


# Initialize runner
runner = CliRunner()

def test_cli_command() -> None:
    """Test whether the method is working by CLI generally"""
    (args, ) = get_data_config("vars", {"predict": ["args"]},
                               path = "./dvc.yaml",
                               is_convert_to_dict = True)
    result = runner.invoke(predict, args) # type: ignore
    assert result.exit_code == 0
