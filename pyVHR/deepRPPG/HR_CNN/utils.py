from scipy.signal import butter, lfilter
import torch
import torch.nn as nn


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def psnr(img, img_g):
    criterionMSE = nn.MSELoss()  # .to(device)
    mse = criterionMSE(img, img_g)

    psnr = 10 * torch.log10(torch.tensor(1) / mse)  # 20 *
    return psnr
