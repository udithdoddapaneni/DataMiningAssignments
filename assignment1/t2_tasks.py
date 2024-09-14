# GiG
from typing import Any


# This function accepts python_dict where all keys are strings
# and the values are strings, integers and complex combination of lists and dictionaries
# It outputs a string that is a valid TOML string
# Note: the python dictionary can be nested (e.g. list of list, dict of dicts etc)
# So design this as a recursive function
# Hint: for nested dictionaries, it is simpler to output as fully named tables.
# For example, if there is a dict inner inside outer,
# then something like
# [outer]
# [outer.inner]
# is the way to go.
# This can be done by using the optional parent_key variable


def t2c_python_dict_to_toml_string(python_dict: dict[str, Any], parent_key: str = "") -> str:
    toml = ""
    key = "\n" if parent_key == "" else f"[{parent_key}]\n"
    toml += key
    keys = list(python_dict)
    for i in keys:
        if isinstance(python_dict[i], dict):
            pkey = i if parent_key == "" else parent_key + "." + i
            toml += t2c_python_dict_to_toml_string(python_dict[i], pkey)
        else:
            key = "\n" if parent_key == "" else f"[{parent_key}]\n"
            value = f'"{python_dict[i]}"' if isinstance(python_dict[i], str) else python_dict[i]
            toml += f"{i} = {value}\n"
    return toml


def t2d_python_dict_to_yaml_string(python_dict: dict[str, Any], indent: int = 0) -> str:
    yaml = ""
    ident = "  " * (indent)
    space = " "
    for e in python_dict:
        if isinstance(python_dict[e], dict):
            yaml += (
                ident + e + ":" + "\n" + t2d_python_dict_to_yaml_string(python_dict[e], indent + 1)
            )
        else:
            if isinstance(python_dict[e], str):
                value = f'"{python_dict[e]}"'
            elif isinstance(python_dict[e], list):
                value = "\n"
                for i in python_dict[e]:
                    v = f'"{i}"' if isinstance(i, str) else f"{i}"
                    value += "  " * (indent + 1) + "- " + v + "\n"
            else:
                value = python_dict[e]
            yaml += ident + e + ":" + space + str(value) + "\n"

    return yaml
