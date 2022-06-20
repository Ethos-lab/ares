import json


def load_config(path: str):
    with open(path, "r") as f:
        config = json.load(f)
    # TODO: validate config file
    return config
