import gc

import numpy as np
import torch


def get_mse(original, compressed):
    if isinstance(original, torch.Tensor):
        original = get_array(original)
    if isinstance(compressed, torch.Tensor):
        compressed = get_array(compressed)

    return np.mean((original - compressed) ** 2)


def get_rmse(original, compressed):
    mse = get_mse(original, compressed)
    return np.sqrt(mse)


def get_psnr(original, compressed):
    if isinstance(original, torch.Tensor):
        original = get_array(original)
    if isinstance(compressed, torch.Tensor):
        compressed = get_array(compressed)

    mse = get_mse(original, compressed)
    if mse == 0:
        return np.inf
    max_pixel = np.abs(original).max()
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def clean_up():
    gc.collect()
    torch.cuda.empty_cache()


def count_weights(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_array(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    else:
        return x
    

def estimate_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return param_size + buffer_size