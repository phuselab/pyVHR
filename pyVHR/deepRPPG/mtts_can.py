import pyVHR
import numpy as np
import tensorflow as tf
from .MTTS_CAN.model import Attention_mask, MTTS_CAN
from scipy.signal import butter
import cv2
from skimage.util import img_as_float
import scipy.io
from scipy.sparse import spdiags
import h5py
import os
import requests

def preprocess_raw_video(frames, fs=30, dim=36):
  """A slightly different version from the original: 
    takes frames as input instead of video path """

  totalFrames = frames.shape[0]
  Xsub = np.zeros((totalFrames, dim, dim, 3), dtype=np.float32)
  i = 0
  t = []
  width = frames.shape[2]
  height = frames.shape[1]
  # Crop each frame size into dim x dim
  for img in frames:
    t.append(1/fs*i)       # current timestamp in milisecond
    img = img[:, int(width/2)-int(height/2 + 1):int(height/2)+int(width/2), :]
    vidLxL = cv2.resize(img_as_float(img), (dim, dim), interpolation = cv2.INTER_AREA)
    vidLxL[vidLxL > 1] = 1
    vidLxL[vidLxL < (1/255)] = 1/255
    Xsub[i, :, :, :] = vidLxL
    i = i + 1
  #import matplotlib.pyplot as plt
  #plt.imshow(Xsub[0])
  #plt.show()

  # Normalized Frames in the motion branch
  normalized_len = len(t) - 1
  dXsub = np.zeros((normalized_len, dim, dim, 3), dtype = np.float32)
  for j in range(normalized_len - 1):
    dXsub[j, :, :, :] = (Xsub[j+1, :, :, :] - Xsub[j, :, :, :]) / (Xsub[j+1, :, :, :] + Xsub[j, :, :, :])
  dXsub = dXsub / np.std(dXsub)
  
  # Normalize raw frames in the apperance branch
  Xsub = Xsub - np.mean(Xsub)
  Xsub = Xsub  / np.std(Xsub)
  Xsub = Xsub[:totalFrames-1, :, :, :]
  
  # Plot an example of data after preprocess
  dXsub = np.concatenate((dXsub, Xsub), axis=3);
  return dXsub

def detrend(signal, Lambda):
    """detrend(signal, Lambda) -> filtered_signal
    This function applies a detrending filter.
    This code is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
    *Parameters*
      ``signal`` (1d numpy array):
        The signal where you want to remove the trend.
      ``Lambda`` (int):
        The smoothing parameter.
    *Returns*
      ``filtered_signal`` (1d numpy array):
        The detrended signal.
    """
    signal_length = signal.shape[0]

    # observation matrix
    H = np.identity(signal_length)

    # second-order difference matrix

    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal

def MTTS_CAN_deep(frames, fs, model_checkpoint=None, batch_size=100, dim=36, img_rows=36, img_cols=36, frame_depth=10, verb=0, filter_pred=False):

  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
     for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)

  if model_checkpoint is None:
    model_checkpoint = pyVHR.__path__[0] +  '/deepRPPG/MTTS_CAN/mtts_can_model.hdf5'
    if not os.path.isfile(model_checkpoint):
      url = "https://github.com/phuselab/pyVHR/raw/master/resources/deepRPPG/mtts_can_model.hdf5"
      print('Downloading MTTS_CAN model...')
      r = requests.get(url, allow_redirects=True)
      open(model_checkpoint, 'wb').write(r.content)   

  # frame preprocessing
  dXsub = preprocess_raw_video(frames, fs=fs, dim=dim)
  dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
  dXsub = dXsub[:dXsub_len, :, :, :]

  # load pretrained model
  model = MTTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
  model.load_weights(model_checkpoint)

  # apply pretrained model
  yptest = model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=verb)

  # filtering
  
  pulse_pred = yptest[0]
  pulse_pred = detrend(np.cumsum(pulse_pred), 100)
  if filter_pred:
    [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    #[b_pulse, a_pulse] = butter(1, [0.65 / fs * 2, 2.5 / fs * 4], btype='bandpass')
    pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))
  return pulse_pred
