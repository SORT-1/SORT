import random
from collections import OrderedDict

import torch


def generate_and_run_invoke_str(api_name, node_input, api_def):
    object_str = None
    node_input = OrderedDict(sorted(node_input.items()))
    for def_str in api_def:
        def_api_name = def_str.split('(')[0].split('.')[-1]
        if api_name == def_api_name:
            object_str = def_str.split('(')[0]
            break

    params = []
    for node_name, node_value in node_input.items():
        params.append(node_value)
    object_str = f'{object_str}(*params)'
    output = eval(object_str)

    return output
