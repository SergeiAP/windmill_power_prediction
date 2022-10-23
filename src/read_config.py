# pylint: disable=missing-module-docstring
# pylint: disable=missing-module-docstring
import logging
import yaml
from yaml.loader import SafeLoader


class ProcessConfig:
    """Class to upload, store and retrieve config"""
    
    def __init__(self, filepath: str) -> None:
        """initialise values

        Args:
            filepath (str): path to config
        """
        self.cofig = self.upload_config(filepath)
        
    def get(self, section: str, columns: list[str] | dict[str, list[str]]) -> tuple:
        """Get data from config

        Args:
            section (str): name of params section
            columns (list[str] | dict[str, list[str]]): selction-names of params 
            required to be extracted from config

        Returns:
            tuple: any params from the section. Num of params = len(columns) or values 
            in dict list
        """
        match columns:
            case list(columns):
                config = tuple(self.cofig[section][col] for col in columns)
            case dict(columns):
                config = tuple(self.cofig[section][key][col]
                               for key in columns
                               for col in columns[key])
        logging.info("Config section `%s`, values %s were retrieved", section, columns)
        return config
    
    @classmethod 
    def upload_config(cls, filepath: str) -> dict:
        """Upload config

        Args:
            filepath (str): path to config

        Returns:
            dict: uploaded config
        """
        with open(filepath, mode="r", encoding="utf-8") as config_file:
            config = yaml.load(config_file, Loader=SafeLoader)
        logging.info("Config from %s was uploaded", filepath)
        return config


def get_data_config(section: str,
                    columns: list[str] | dict[str, list[str]], 
                    path: str = "./src/config.yml",
                    is_convert_to_dict: bool = False) -> tuple:
    """Read config.yml and extract required params from specific section
    Args:
        section (str): name of params section
        columns (list[str] | dict[str, list[str]]): selction-names of params required 
        to be extracted from config
        path (str): path to the cofig. Default to "./src/config.yml".
        is_convert_to_dict (bool): whether to conver list into dict or not 
        (applicable for vars in dvc.yaml)
    Returns:
        tuple: any params from the section. Num of params = len(columns) or values 
        in dict list
    """
    with open(path, mode="r", encoding="utf-8") as config_file:
        config = yaml.load(config_file, Loader=SafeLoader)[section]
        match columns:
            case list(columns):
                config = tuple(config[col] for col in columns)
            case dict(columns):
                if is_convert_to_dict:
                    config = {k:v for elem in config for k,v in elem.items()}
                config = tuple(config[key][col]
                               for key in columns
                               for col in columns[key])
    return config
