# Utils for initialization

# Libraries
import torch
import math

init_weight_scale = 1e-6


def kaiming_uniform_zeros_(m): return torch.nn.init.kaiming_uniform_(m, a=math.sqrt((6. / init_weight_scale**2) - 1))


def kaiming_normal_zeros_(m): return torch.nn.init.kaiming_normal_(m, a=math.sqrt((6. / init_weight_scale**2) - 1))


def init_bias_uniform_zeros(m, weight):
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
    if fan_in > 0:
        bound = init_weight_scale / math.sqrt(fan_in)
        torch.nn.init.uniform_(m, -bound, bound)
    else:
        torch.nn.init.zeros_(m)


def init_bias_normal_zeros(m, weight):
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
    if fan_in > 0:
        std = init_weight_scale / math.sqrt(fan_in)
        torch.nn.init.normal_(m, mean=0.0, std=std)
    else:
        torch.nn.init.zeros_(m)


def init_bias_uniform_constant(m, weight, val=1.0):
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
    if fan_in > 0:
        bound = init_weight_scale / math.sqrt(fan_in)
        torch.nn.init.uniform_(m, val-bound, val+bound)
    else:
        torch.nn.init.constant_(m, val)


def init_bias_normal_constant(m, weight, val=1.0):
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
    if fan_in > 0:
        std = init_weight_scale / math.sqrt(fan_in)
        torch.nn.init.normal_(m, mean=val, std=std)
    else:
        torch.nn.init.constant_(m, val)
