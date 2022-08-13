import os
from pathlib import Path
from typing import Any, Dict

import toml

ENVIRONMENT = os.getenv("ENVIRONMENT_EXECUTION", "")
LOCAL_MODE = ENVIRONMENT in ["LOCAL"]

map_env_config_folder = {
    "DEV": "DEV",
    "LOCAL": "LOCAL",
}

map_config_types = {
    "settings": "settings.toml",
    "model": "model.toml",
    "business": "business.toml",
    "extraction": "extraction.toml",
}


class Config(dict):
    """Class containing the different types of configuration.
        The object's underlying dictionary is built from a toml file.

    :Parameters:
        - `conf_type` (str)
            Specifies which type of config to load. Possible values are:
            * business
            * settings
            * model

    :Attributes:
            - `type` (str)
                holds the value of conf_type passed to constructor
    """

    def __init__(self, conf_type: str) -> None:
        self.type = conf_type
        self._path = self.get_config_path(conf_type)
        dico_conf = self.load_toml_file(self._path)
        super().__init__(dico_conf)

    @staticmethod
    def get_config_path(conf_type: str) -> Path:
        """Return the path to configuration.

        Configuration file must be present in `config/` directory at project root.

        :Parameters:
        :Return:
            - Path object to configuration file
        """

        # Get the current directory (where this file resides)
        real_dir = os.path.dirname(os.path.realpath(__file__))
        if ENVIRONMENT and ENVIRONMENT in map_env_config_folder.keys():
            path_to_conf = os.path.join(
                real_dir,
                "..",
                "config",
                map_env_config_folder[ENVIRONMENT],
                map_config_types[conf_type],
            )
        else:
            raise ValueError(
                "The environment variable for the execution was not recognized. Please "
                "define the os variable ENVIRONMENT_EXECUTION with one of the values "
                f"permitted: {list(map_env_config_folder.keys())}."
            )

        return Path(path_to_conf)

    @staticmethod
    def load_toml_file(config_path: Path) -> Dict[str, Any]:
        """Parse and return a toml configuration file as a dict.

        :Parameters:
            - `config_path` (Path)
                Path object to configuration file
        :Return:
            - dict containing the configuration
        """

        with open(config_path) as file_content:
            # force conversion to Dict (else will be a MutableMapping)
            config = dict(toml.load(file_content))
        return config


if __name__ == "__main__":
    cfg_setting = Config("settings")
    cfg_model = Config("model")
    cfg_business = Config("business")
    pass
