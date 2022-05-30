# pylint: disable=missing-module-docstring
import yaml
from yaml.loader import SafeLoader


def get_data_config(section: str, columns: list[str]) -> tuple:
    """Read config.yml and extract required params from specific section
    Args:
        section (str): name of params section
        columns (list[str]): columns-names of params required to be extracted 
        from config, `data` section
    Returns:
        tuple: any params from the section. Num of params = len(columns)
    """
    with open("./src/config.yml", mode="r", encoding="utf-8") as config_file:
        config = yaml.load(config_file, Loader=SafeLoader)
        config = tuple(config[section][col] for col in columns)
    return config
