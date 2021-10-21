from numba import cuda
import torch
import os


def cuda_info():
    if torch.cuda.is_available():
        print("# CUDA devices: ", torch.cuda.device_count())
        for e in range(torch.cuda.device_count()):
            print("# device number ", e, ": ", torch.cuda.get_device_name(e))


def select_cuda_device(n):
    torch.cuda.device(n)
    cuda.select_device(n)
