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

def akurtosis(sample, dt, win=3.0, rtmemory_list=None):
    from obspy.realtime.rtmemory import RtMemory
    # deal with case of empty trace
    if not rtmemory_list:
        rtmemory_list = [RtMemory(), RtMemory(), RtMemory()]
    if np.size(sample) < 1:
        return sample

    npts = len(sample)

    c_1 = dt / float(win)
    a1 = 1.0 - c_1
    c_2 = (1.0 - a1 * a1) / 2.0
    bias = -3 * c_1 - 3.0

    kappa4 = np.empty(npts, sample.dtype)

    rtmemory_mu1 = rtmemory_list[0]
    rtmemory_mu2 = rtmemory_list[1]
    rtmemory_k4_bar = rtmemory_list[2]

    if not rtmemory_mu1.initialized:
        memory_size_input = 1
        memory_size_output = 0
        rtmemory_mu1.initialize(sample.dtype, memory_size_input,
                                memory_size_output, 0, 0)

    if not rtmemory_mu2.initialized:
        memory_size_input = 1
        memory_size_output = 0
        rtmemory_mu2.initialize(sample.dtype, memory_size_input,
                                memory_size_output, 1, 0)

    if not rtmemory_k4_bar.initialized:
        memory_size_input = 1
        memory_size_output = 0
        rtmemory_k4_bar.initialize(sample.dtype, memory_size_input,
                                   memory_size_output, 0, 0)

    mu1_last = rtmemory_mu1.input[0]
    mu2_last = rtmemory_mu2.input[0]
    k4_bar_last = rtmemory_k4_bar.input[0]

    # do recursive kurtosis
    for i in range(npts):
        mu1 = a1 * mu1_last + c_1 * sample[i]
        dx2 = (sample[i] - mu1_last) * (sample[i] - mu1_last)
        mu2 = a1 * mu2_last + c_2 * dx2
        dx2 = dx2 / mu2_last
        k4_bar = (1 + c_1 - 2 * c_1 * dx2) * k4_bar_last + c_1 * dx2 * dx2
        kappa4[i] = k4_bar + bias
        mu1_last = mu1
        mu2_last = mu2
        k4_bar_last = k4_bar

    rtmemory_mu1.input[0] = mu1_last
    rtmemory_mu2.input[0] = mu2_last
    rtmemory_k4_bar.input[0] = k4_bar_last

    return kappa4