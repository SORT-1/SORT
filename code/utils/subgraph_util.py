import numpy as np
import torch


def convert_to_cpu(d):
    for key in d.keys():
        val = d[key]
        if type(val) == torch.Tensor:
            d[key] = val.cpu()


def convert_to_gpu(d):
    for key in d.keys():
        val = d[key]
        if type(val) == torch.Tensor:
            d[key] = val.cuda()

