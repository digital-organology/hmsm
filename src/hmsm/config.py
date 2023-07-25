# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import json
import logging
import re

import pkg_resources


def get_config(config: str, method: str) -> dict:
    """Find and load configuration data

    This function will do one of the following three things, depending on the provided config argument:
    If the argument seems to be a filename (ends in .json) it will try to open that file and read the configuration from it.
    If the argument looks like a json string it will parse that string and test if it contains a valid configuration profile.
    In all other cases it will treat the argument as the name of a configuration profile to load from the package internal config file.

    Args:
        config (str): Either: The name of an internal configuration preset, the name of a json file with configuration data or a json string containing json data
        method (str): Processing method that needs to be configured

    Raises:
        KeyError: Will be raised if the requested profile is missing from the internal configuration
        FileNotFoundError: Will be raised if a filepath was provided but the file can not be found
        ValueError: Will be raised if the specified profile is found but not suitable for the requested processing method


    Returns:
        dict: Read and validated configuration data
    """

    if re.match("\.json$", config):
        logging.info(
            "Provided config looks like a filename, trying to read configuration from that file"
        )
        try:
            with open(config, "r", encoding="UTF-8") as f:
                config_data = json.load(f)
        except FileNotFoundError:
            logging.error(f"Failed to open file '{config}'")
            raise
        logging.info(f"Read configuration from file {config}")
    elif re.match("^\{.*\}$", config.strip()):
        logging.info(
            "Provided config looks like a json string, trying to read configuration from the string"
        )
        try:
            config_data = json.loads(config)
        except ValueError:
            logging.error("Failed to parse json string")
            raise
        logging.info("Parsed configuration from json string")
    else:
        logging.info("Trying to find configuration profile in internal config file")
        try:
            file = pkg_resources.resource_filename(__name__, "data/config.json")
            with open(file, "r", encoding="UTF-8") as f:
                config_data = json.load(f)

            config_data = config_data[config]

            if config_data["method"] == method:
                logging.info("Configuration read from internal config file")
            else:
                raise ValueError(
                    "Matching configuration profile found but unsuitable for processing method to be executed"
                )
        except KeyError:
            logging.error(
                "No matching configuration data found in internal config file"
            )
            raise
        except FileNotFoundError:
            logging.error("Failed to open internal config file, something is broken")
            raise
        except ValueError:
            raise

    if validate_config(config_data, method):
        logging.info("Configuration loaded and verified")
        return config_data

    raise ValueError("Failed to validate the given configuration")


def validate_config(config: dict, method: str) -> bool:
    """Checks if the given configuration data is sufficient to execute the requested operation

    Args:
        config (dict): Configuration data
        method (str): Digitization method to apply

    Raises:
        ValueError: Will be raised if the method is unsupported

    Returns:
        bool: True if the configuration is valid, False otherwise
    """
    if method == "cluster":
        return _validate_cluster_config(config)
    elif method == "roll":
        return _validate_roll_config(config)

    raise ValueError(f"method '{method}' currently not implemented")


def _validate_cluster_config(config: dict) -> bool:
    """Checks if all required parameters for clustering based midi digitization are set

    Args:
        config (dict): Configuration parameters

    Returns:
        bool: True if the configuration is valid, False otherwise
    """
    required_keys = {"radius_inner", "first_track", "track_mapping"}
    if not required_keys <= config.keys():
        logging.error(
            f"Configuration misses the following required values: {required_keys - config.keys()}"
        )
        return False

    # TODO: Implement type/range checking
    return True


def _validate_roll_config(config: dict) -> bool:
    """Checks if all required parameters for piano roll digitization are set

    Args:
        config (dict): Configuration parameters

    Returns:
        bool: True if the configuration is valid, False otherwise
    """
    required_keys = {"track_measurements", "roll_width_mm"}
    if not required_keys <= config.keys():
        logging.error(
            f"Configuration misses the following required values: {required_keys - config.keys()}"
        )
        return False

    return True
