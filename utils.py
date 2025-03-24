import os
import torch
import yaml
import math
import torch.nn as nn
from types import SimpleNamespace

def get_optimizer(model, config):
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    net_params = [p for n, p in param_dict.items() if p.dim() >= 2 and 'embedding' not in n]
    embed_params = [p for n, p in param_dict.items() if p.dim() >= 2 and 'embedding' in n]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': net_params, 'weight_decay': config.net_weight_decay},
        {'params': embed_params, 'weight_decay': config.embedding_weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate, 
                                  betas=(0.9, 0.999), eps=1e-8)
    return optimizer

def get_activation(activation):
    if isinstance(activation, str):
        if activation.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif activation.lower() == 'linear':
            act_layer = nn.Identity
        elif activation.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif activation.lower() == 'silu':
            act_layer = nn.SiLU()
        elif activation.lower() == 'gelu':
            act_layer = nn.GELU()
    else:
        raise NotImplementedError

    return act_layer

def load_config(args):
    dataset_config_path = args.dataset_config_path
    model_config_path = args.model_config_path

    try:
        with open(dataset_config_path, "r") as f:
            dataset_config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        dataset_config = {}

    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f) or {}

    config = {**dataset_config, **model_config}

    return SimpleNamespace(**config)

def get_lr_lambda(it, warmup_steps, max_steps, max_lr, min_lr):
    # warm up
    if it < warmup_steps:
        return it / warmup_steps
    # after cosine decay
    if it > max_steps:
        return min_lr / max_lr
    # cosine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (min_lr + coeff * (max_lr - min_lr)) / max_lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
