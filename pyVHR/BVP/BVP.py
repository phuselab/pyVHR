import cupy
import numpy as np
import torch
import sys

"""
This module contains methods for trasforming an input signal
in a BVP signal using a rPPG method (see pyVHR.BVP.methods).
"""

def signals_to_bvps_cuda(sig, cupy_method, params={}):
    """
    Transform an input RGB signal in a BVP signal using a rPPG method (see pyVHR.BVP.methods).
    This method must use cupy and executes on GPU. You can pass also non-RGB signal but the method used must handle its shape.

    Args:
        sig (float32 ndarray): RGB Signal as float32 ndarray with shape  [num_estimators, rgb_channels, num_frames].
            You can pass also a generic signal but the method used must handle its shape and type.
        cupy_method: a method that comply with the fucntion signature documented in pyVHR.BVP.methods. This method must use Cupy.
        params (dict): dictionary of usefull parameters that will be passed to the method.
    
    Returns:
        float32 ndarray: BVP signal as float32 ndarray with shape [num_estimators, num_frames].
    """
    if sig.shape[0] == 0:
        return np.zeros((0, sig.shape[2]), dtype=sig.dtype)
    gpu_sig = cupy.asarray(sig)
    if len(params) > 0:
        bvps = cupy_method(gpu_sig, **params)
    else:
        bvps = cupy_method(gpu_sig)
    r_bvps = cupy.asnumpy(bvps)
    gpu_sig = None
    bvps = None
    return r_bvps


def signals_to_bvps_torch(sig, torch_method, params={}):
    """
    Transform an input RGB signal in a BVP signal using a rPPG 
    method (see pyVHR.BVP.methods).
    This method must use Torch and execute on GPU/CPU.
    You can pass also non-RGB signal but the method used must handle its shape.

    Args:
        sig (float32 ndarray): RGB Signal as float32 ndarray with shape  [num_estimators, rgb_channels, num_frames].
            You can pass also a generic signal but the method used must handle its shape and type.
        torch_method: a method that comply with the fucntion signature documented 
            in pyVHR.BVP.methods. This method must use Torch.
        params (dict): dictionary of usefull parameters that will be passed to the method.
    
    Returns:
        float32 ndarray: BVP signal as float32 ndarray with shape [num_estimators, num_frames].
    """
    if sig.shape[0] == 0:
        return np.zeros((0, sig.shape[2]), dtype=sig.dtype)
    cpu_sig = torch.from_numpy(sig)
    if len(params) > 0:
        bvps = torch_method(cpu_sig, **params)
    else:
        bvps = torch_method(cpu_sig)
    return bvps.numpy()


def signals_to_bvps_cpu(sig, cpu_method, params={}):
    """
    Transform an input RGB signal in a BVP signal using a rPPG 
    method (see pyVHR.BVP.methods).
    This method must use and execute on CPU.
    You can pass also non-RGB signal but the method used must handle its shape.

    Args:
        sig (float32 ndarray): RGB Signal as float32 ndarray with shape  [num_estimators, rgb_channels, num_frames].
            You can pass also a generic signal but the method used must handle its shape and type.
        cpu_method: a method that comply with the fucntion signature documented 
            in pyVHR.BVP.methods. This method must use Numpy.
        params (dict): dictionary of usefull parameters that will be passed to the method.
    
    Returns:
        float32 ndarray: BVP signal as float32 ndarray with shape [num_estimators, num_frames].
    """
    if sig.shape[0] == 0:
        return np.zeros((0, sig.shape[2]), dtype=sig.dtype)
    cpu_sig = np.array(sig)
    if len(params) > 0:
        bvps = cpu_method(cpu_sig, **params)
    else:
        bvps = cpu_method(cpu_sig)
    return bvps


def RGB_sig_to_BVP(windowed_sig, fps, device_type=None, method=None, params={}):
    """
    Transform an input RGB windowed signal in a BVP windowed signal using a rPPG method (see pyVHR.BVP.methods).
    You can pass also non-RGB signal but the method used must handle its shape.

    Args:
        windowed_sig (list): RGB windowed signal as a list of length num_windows of np.ndarray with shape [num_estimators, rgb_channels, num_frames].
        fps (float): frames per seconds. You can pass also a generic signal but the method used must handle its shape and type.
        device_type (str): the chosen rPPG method run on GPU ('cuda'), or CPU ('cpu', 'torch').
        method: a method that comply with the fucntion signature documented 
            in pyVHR.BVP.methods. This method must use Numpy if the 'device_type' is 'cpu', Torch if the 'device_type' is 'torch', and Cupy 
            if the 'device_type' is 'cuda'.
        params(dict): dictionary of usefull parameters that will be passed to the method. If the method needs fps you can set {'fps':'adaptive'}
            in the dictionary to use the 'fps' input variable.

    Returns:
        a list of lenght num_windows of BVP signals as np.ndarray with shape [num_estimators, num_frames];
        if no BVP can be found in a window, then the np.ndarray has num_estimators == 0.
    """
    if device_type != 'cuda' and device_type != 'cpu' and device_type != 'torch':
        print("[ERROR]: invalid device_type!")
        return []

    if 'fps' in params and params['fps'] == 'adaptive':
        params['fps'] = np.float32(fps)

    bvps = []
    for sig in windowed_sig:
        copy_signal = np.copy(sig)
        bvp = np.zeros((0, 1), dtype=np.float32)
        if device_type == 'cpu':
            bvp = signals_to_bvps_cpu(
                copy_signal, method, params)
        elif device_type == 'torch':
            bvp = signals_to_bvps_torch(
                copy_signal, method, params)
        elif device_type == 'cuda':
            bvp = signals_to_bvps_cuda(
                copy_signal, method, params)
        bvps.append(bvp)
    return bvps


def concatenate_BVPs(list_of_BVPs):
    """
    Join a list of windowed BVPs. There must be the same number of windows, and each one must have the same number of frames.

    Args:
        list_of_BVPs (list): a list of windowed BVPs each one defined as a list of lenght num_windows of BVP signals as np.ndarray
            with shape [num_estimators, num_frames]. Remember that concatenation is possible only if 'num_frames'
            is the same for each BVP of the nth window.
            
    Returns:
        a list of lenght num_windows of BVP signals as np.ndarray with shape [total_num_estimators, num_frames];
        if no BVP can be found in a window, then the np.ndarray has total_num_estimators == 0. 
        For example: given the BVP windows of shape [10, 200] and [20,200], the concatenated window will have shape [30,200];
        If an exception is thrown for any reason, the function returns 0.
    """
    if len(list_of_BVPs) <= 1:
        return 0
    
    first_window_len = len(list_of_BVPs[0])

    for item in list_of_BVPs:
        if first_window_len != len(item):
            return 0

    try:
        concatenated_BVPs = []
        for i in range(first_window_len):
            concatenated_BVPs.append(np.concatenate(tuple([e[i] for e in list_of_BVPs]), axis=0))
        return concatenated_BVPs
    except ValueError as e:
        print(e)
        return 0