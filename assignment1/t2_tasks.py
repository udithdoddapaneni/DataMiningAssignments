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
        elif isinstance(python_dict[i], list) and len(python_dict[i]) > 0 and\
        isinstance(python_dict[i][0], dict):
            # special case: toml table
            value = f"[[{i}]]"
            for j in python_dict[i]:
                value += t2c_python_dict_to_toml_string(j)
            toml += value
        else:
            key = "\n" if parent_key == "" else f"[{parent_key}]\n"
            value = f'"{python_dict[i]}"' if isinstance(python_dict[i], str) else python_dict[i]
            toml += f"{i} = {value}\n"
    return toml

def t2d_python_dict_to_yaml_string(python_dict: dict[str, Any], indent: int = 0) -> str:
    yaml = ""
    ident = "  " * (indent)
    space = " "
    def parse_list(li: list, spacing:int = 0) -> str:
        value = ""
        first = True
        for i in li:
            if isinstance(i, str):
                v = f'"{i}"'
            elif isinstance(i, dict):
                v = t2d_python_dict_to_yaml_string(i, indent)
            elif isinstance(i, list):
                v = parse_list(i, spacing+1)
            else:
                v = str(i)
            if not isinstance(i, list):
                if first:
                    value += "  " * (indent + 1) + "- " * spacing + "- " + v + "\n"
                    first = False
                else:
                    value += "  " * (indent + 1) + "  " * spacing + "- " +  v + "\n"
            else:
                value += v
        return value
    for e in python_dict:
        if isinstance(python_dict[e], dict):
            yaml += (
                ident + e + ":" + "\n" + t2d_python_dict_to_yaml_string(python_dict[e], indent + 1)
            )
        else:
            if isinstance(python_dict[e], str):
                value = f'"{python_dict[e]}"'
            elif isinstance(python_dict[e], list):
                value = "\n" + parse_list(python_dict[e])
            else:
                value = python_dict[e]
            yaml += ident + e + ":" + space + str(value) + "\n"

    return yaml
