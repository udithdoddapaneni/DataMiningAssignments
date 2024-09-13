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
    if parent_key == "":
        key = "\n"
    else:
        key = f"[{parent_key}]\n"
    toml += key
    keys = []
    for i in python_dict:
        if type(python_dict[i]) != dict:
            keys.append(i)

    for i in python_dict:
        if type(python_dict[i]) == dict:
            keys.append(i)

    for i in keys:
        if type(python_dict[i]) == dict:
            if parent_key == "":
                pkey = i
            else:
                pkey = parent_key + "." + i
            toml += t2c_python_dict_to_toml_string(python_dict[i], pkey)
        else:
            if parent_key == "":
                key = "\n"
            else:
                key = f"[{parent_key}]\n"
            
            if type(python_dict[i]) == str:
                value = f'"{python_dict[i]}"'
            else:
                value = python_dict[i]
            toml += f"{i} = {value}\n"
    return toml

def t2d_python_dict_to_yaml_string(python_dict: dict[str, Any], indent: int = 0) -> str:
    yaml = ""
    ident = "  "*(indent)
    space = " "
    for e in python_dict:
        if type(python_dict[e]) == dict:
            yaml += ident + e + ":" + "\n" + t2d_python_dict_to_yaml_string(python_dict[e], indent+1)
        else:
            if type(python_dict[e]) == str:
                value = f'"{python_dict[e]}"'
            elif type(python_dict[e]) == list:
                value = "\n"
                for i in python_dict[e]:
                    if type(i) == str:
                        v = f'"{i}"'
                    else:
                        v = f"{i}"
                    value += "  "*(indent+1) + "- " + v + "\n"
            else:
                value = python_dict[e]            
            yaml += ident + e + ":" + space + str(value) + "\n"

    return yaml
