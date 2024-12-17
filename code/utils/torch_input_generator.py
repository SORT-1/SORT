import math
import pickle
import random

import numpy as np
import torch
from torch import nn

import torch

from utils.ErrorStartException import ErrorStartException


def generate_linear(param_name):
    res = None
    if param_name == "input":
        res = "parameter:0"
    elif param_name == "weight":
        res = "parameter:1"
    elif param_name == "bias":
        res = "parameter:2"
    return res


def generate_uniform_value(val_min, val_max, shape, param_type):
    if param_type == "torch.float64":
        x = np.random.uniform(val_min, val_max, shape)
        x = torch.DoubleTensor(x)
    elif param_type == "torch.float32":
        x = np.random.uniform(val_min, val_max, shape)
        x = torch.FloatTensor(x)
    elif param_type == "torch.bool":
        x = np.random.choice([val_min, val_max], shape)
        x = torch.BoolTensor(x)
    elif param_type == "torch.int64":
        x = np.random.randint(val_min, val_max, shape)
        x = torch.LongTensor(x)
    elif param_type == "torch.int32":
        x = np.random.randint(val_min, val_max, shape)
        x = torch.IntTensor(x)

    return x


def generate_true_input(name, config_data):
    params = {}
    configs = []

    for config_ori in config_data:
        method = config_ori

        if name == method.split(".")[-1]:
            configs = config_data[config_ori].copy()

    while True:
        index = random.randint(0, len(configs) - 1)
        config = configs[index]

        if "torch.uint8" not in str(config):
            break

    for param_name, param_config_ori in config.items():
        if param_name.split(':')[0] == "parameter":
            param_config = param_config_ori
            if isinstance(param_config, dict):
                params[param_name] = generate_uniform_value(param_config["min"], param_config["max"],
                                                            param_config["shape"], param_config["dtype"])
            else:
                params[param_name] = param_config
        elif name == "linear" and param_name.split(':')[0] != "parameter":
            param_config = param_config_ori
            param_name = generate_linear(param_name)
            if param_name is not None:
                params[param_name] = generate_uniform_value(param_config["min"], param_config["max"],
                                                            param_config["shape"], param_config["dtype"])

    return params


def generate_fix_input(name, config_data, input):
    params = {}
    configs = []

    for config_ori in config_data:
        method = config_ori

        if name == method.split(".")[-1]:
            configs = config_data[config_ori].copy()

    true_configs = []
    for config in configs:
        for param_name, param_config_ori in config.items():
            if param_name.split(':')[0] == "parameter":
                if param_name.split(':')[1] == "0":
                    param_config = param_config_ori
                    if list(input.shape) == list(param_config["shape"]) and str(input.dtype) == param_config['dtype']:
                        true_configs.append(config)
                        params["parameter:0"] = input


    if len(true_configs) == 0:
        raise ErrorStartException("Error")
    config = true_configs[random.randint(0, len(true_configs) - 1)]
    for param_name, param_config_ori in config.items():
        if param_name.split(':')[0] == "parameter":
            if param_name.split(':')[1] != "0":
                param_config = param_config_ori
                if isinstance(param_config, dict):
                    if param_config['dtype'] != str(input.dtype):
                        param_config["dtype"] = str(input.dtype)
                    params[param_name] = generate_uniform_value(param_config["min"], param_config["max"],
                                                                param_config["shape"], param_config["dtype"])
                else:
                    params[param_name] = param_config
    return params


def change_pre_api_output(params, input):
    res_params = params.copy()
    res_params["parameter:0"] = input

    return res_params
