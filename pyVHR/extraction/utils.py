from numba import prange, njit
import numpy as np
import cv2
from numba import prange, njit
import numpy as np
from scipy.signal import welch, butter, filtfilt, iirnotch, freqz
from sklearn.decomposition import PCA
from scipy.stats import iqr, median_abs_deviation

class MotionAnalysis():
  """
    Extraction MediaPipe landmarks for motion analysis
  """

  def __init__(self, sig_extractor, winsize, fps, stride=1, landmks=None, Q_notch=10):
    self.winsize = winsize
    self.fps = fps
    # Q factor of notch filter
    self.Q_notch = Q_notch
    self.stride = stride
    # selected MediaPipe landmarks
    self.select_landmarks(landmks)
    # all MediaPipe landmarks
    self.landmarks = sig_extractor.get_landmarks() 
    # lanmk IDs effectively used within filter 
    self.landmarks_idx = sig_extractor.ldmks 
    # shapes of cropped frames
    self.shapes = sig_extractor.get_cropped_skin_im_shapes() 
    # win_landmks: lanmks in a win, win_x: x components, win_y: y componets, times: win times
    self.movements_windowing()
      
  def select_landmarks(self, landmks=None):
    """
    Landmark selection from MediaPipe for motion analysis

      Args:
        landmarks (float32 ndarray): ndarray with shape [num_frames, num_estimators, 2-coords].
        shapes (float32 ndarray)   : ndarray with shape [2-coords, num_frames]
        wsize (float)              : window size in seconds.
        stride (float)             : stride between overlapping windows in seconds.
        fps (float)                : frames per seconds.

    """
    # default landmks selected from front to nose
    if landmks is None:
      self.landmks_selected = [10, 151, 6, 197, 5, 4]  
    else:
       self.landmks_selected = landmks

  def movements_windowing(self):
    """
    Calculation of overlapping windows of landmarks coordinates.

    Uses:
        landmarks (float32 ndarray): ndarray with shape [num_frames, num_estimators, 2-coords].
        shapes (float32 ndarray)   : ndarray with shape [2-coords, num_frames]
        wsize (float)              : window size in seconds.
        stride (float)             : stride between overlapping windows in seconds.
        fps (float)                : frames per seconds.

    Provides:
        A list of ndarray (float32) with shape [wsize, num_estimators, 2-coords]
        A list of ndarray (float32) with shape [wsize]
        A list of ndarray (float32) with shape [wsize]
        and an array (float32) of times in seconds (win centers)
    """
    N = self.landmarks.shape[0]
    block_idx, timesLmks = sliding_straded_win_idx(N, self.winsize, self.stride, self.fps)
    lmks_xy = []
    win_x = []
    win_y = []
    for e in block_idx:
      st_frame = int(e[0])
      end_frame = int(e[-1])
      coords = np.copy(self.landmarks[st_frame: end_frame+1])
      x_coords = np.copy(self.shapes[1][st_frame: end_frame+1])
      y_coords = np.copy(self.shapes[0][st_frame: end_frame+1])
      lmks_xy.append(coords)
      win_x.append(x_coords)
      win_y.append(y_coords)
    self.win_landmks = np.array(lmks_xy)
    self.win_x = np.array(win_x)
    self.win_y = np.array(win_y)
    self.timesLmks = timesLmks

  def get_win_motion_filter_old(self, win):
    # loop on landmarks   
    pos = [self.landmarks_idx.index(i) for i in self.landmks_selected if i in self.landmarks_idx]
    X_cords = self.win_landmks[win][:,pos,0]     
    Y_cords = self.win_landmks[win][:,pos,1]
    X_cords = X_cords.T     # shape: (#lanmks, winsize) 
    Y_cords = Y_cords.T     # shape: (#lanmks, winsize) 
    Z_cords = np.expand_dims(self.win_x[win], axis=0)

    Wx, Hx, Ex = self.mov_notch(X_cords)
    Wy, Hy, Ey = self.mov_notch(Y_cords)
    Wz, Hz, Ez = self.mov_notch(Z_cords)

    return Hx, Hy, Hz, Ex, Ey, Ez

  def mov_notch(self, coords):
    
    # coords filtering
    b, a = butter(6, Wn=[.65,4.0], fs=self.fps, btype='bandpass')
    coords = filtfilt(b, a, coords, axis=1)
    F, P = Welch(coords, self.fps)  # shape: (#lanmks, winsize) 
    P = P.T
    pca = PCA(n_components=1)   # PCA
    pca.fit(P)
    P = pca.fit_transform(P).T
    P = P.squeeze()

    # energy 
    P_min = np.min([np.min(P), 0])
    En = np.sum(P-P_min)/(P.shape[0])
    
    # movement notch
    idx = np.argmax(P) 
    P_max = P[idx]
    F_max = F[idx]
    b, a = iirnotch(F_max/60.0, self.Q_notch, self.fps)
    W, H = freqz(b, a, worN=1025,  fs=self.fps)
    band = np.argwhere((W > 0.65) & (W < 4.0)).flatten()
    W_notch = 60.0*W[band]
    H_notch = np.abs(H[band])
    
    #plt.plot(F, P)
    #plt.show()
    #plt.plot(W_notch, H_notch)
    #plt.show()

    return W_notch, H_notch, En


  def get_win_motion_filter(self, win):
    # loop on landmarks   
    pos = [self.landmarks_idx.index(i) for i in self.landmks_selected if i in self.landmarks_idx]
    X_cords = self.win_landmks[win][:,pos,0]     
    Y_cords = self.win_landmks[win][:,pos,1]
    X_cords = X_cords.T     # shape: (#lanmks, winsize) 
    Y_cords = Y_cords.T     # shape: (#lanmks, winsize) 
    Z_cords = np.expand_dims(self.win_x[win], axis=0)

    # filtering
    b, a = butter(6, Wn=[.65,4.0], fs=self.fps, btype='bandpass')

    #----- X -----
    X_cords = filtfilt(b, a, X_cords, axis=1)
    _, XPow = Welch(X_cords, self.fps)  # shape: (#lanmks, winsize) 
    XPow = XPow.T
    pcax = PCA(n_components=1)   # PCA
    pcax.fit(XPow)
    XPow = pcax.fit_transform(XPow).T
    XPmov = XPow.squeeze()
    # energy
    XPmov_min = np.min([np.min(XPmov), 0])
    XEnergy = np.sum(XPmov-XPmov_min)/(XPmov.shape[0])
    XFmov = 1 - XPmov/np.max(XPmov) # X filter 

    #-----Y-----
    Y_cords = filtfilt(b, a, Y_cords, axis=1)
    _, YPow = Welch(Y_cords, self.fps)  # shape: (#lanmks, winsize) 
    YPow = YPow.T
    pcay = PCA(n_components=1)  #PCA
    pcay.fit(YPow)
    YPow = pcay.fit_transform(YPow).T
    YPmov = YPow.squeeze()
    # energy
    YPmov_min = np.min([np.min(YPmov), 0])
    YEnergy = np.sum(YPmov - YPmov_min) / (YPmov.shape[0])
    YFmov = 1 - YPmov/np.max(YPmov)  # Y filter

    #-----Z-----
    Z_cords = filtfilt(b, a, Z_cords, axis=1)    
    _, ZPow = Welch(Z_cords, self.fps)  
    ZPow = ZPow.T
    ZPmov = ZPow.squeeze()  
    ZEnergy = np.sum(ZPmov) / (ZPmov.shape[0])
    ZFmov = 1 - ZPmov/np.max(ZPmov) # Z filter    

    return XFmov, YFmov, ZFmov, XEnergy, YEnergy, ZEnergy

class MagicLandmarks():
    """
    This class contains usefull lists of landmarks identification numbers.
    """
    high_prio_forehead = [10, 67, 69, 104, 108, 109, 151, 299, 337, 338]
    high_prio_nose = [3, 4, 5, 6, 45, 51, 115, 122, 131, 134, 142, 174, 195, 196, 197, 198,
                      209, 217, 220, 236, 248, 275, 277, 281, 360, 363, 399, 419, 420, 429, 437, 440, 456]
    high_prio_left_cheek = [36, 47, 50, 100, 101, 116, 117,
                            118, 119, 123, 126, 147, 187, 203, 205, 206, 207, 216]
    high_prio_right_cheek = [266, 280, 329, 330, 346, 347,
                             347, 348, 355, 371, 411, 423, 425, 426, 427, 436]

    mid_prio_forehead = [8, 9, 21, 68, 103, 251,
                         284, 297, 298, 301, 332, 333, 372, 383]
    mid_prio_nose = [1, 44, 49, 114, 120, 121, 128, 168, 188, 351, 358, 412]
    mid_prio_left_cheek = [34, 111, 137, 156, 177, 192, 213, 227, 234]
    mid_prio_right_cheek = [340, 345, 352, 361, 454]
    mid_prio_chin = [135, 138, 169, 170, 199, 208, 210, 211,
                     214, 262, 288, 416, 428, 430, 431, 432, 433, 434]
    mid_prio_mouth = [92, 164, 165, 167, 186, 212, 322, 391, 393, 410]
    # more specific areas
    forehead_left = [21, 71, 68, 54, 103, 104, 63, 70,
                     53, 52, 65, 107, 66, 108, 69, 67, 109, 105]
    forehead_center = [10, 151, 9, 8, 107, 336, 285, 55, 8]
    forehoead_right = [338, 337, 336, 296, 285, 295, 282,
                       334, 293, 301, 251, 298, 333, 299, 297, 332, 284]
    eye_right = [283, 300, 368, 353, 264, 372, 454, 340, 448,
                 450, 452, 464, 417, 441, 444, 282, 276, 446, 368]
    eye_left = [127, 234, 34, 139, 70, 53, 124,
                35, 111, 228, 230, 121, 244, 189, 222, 143]
    nose = [193, 417, 168, 188, 6, 412, 197, 174, 399, 456,
            195, 236, 131, 51, 281, 360, 440, 4, 220, 219, 305]
    mounth_up = [186, 92, 167, 393, 322, 410, 287, 39, 269, 61, 164]
    mounth_down = [43, 106, 83, 18, 406, 335, 273, 424, 313, 194, 204]
    chin = [204, 170, 140, 194, 201, 171, 175,
            200, 418, 396, 369, 421, 431, 379, 424]
    cheek_left_bottom = [215, 138, 135, 210, 212, 57, 216, 207, 192]
    cheek_right_bottom = [435, 427, 416, 364,
                          394, 422, 287, 410, 434, 436]
    cheek_left_top = [116, 111, 117, 118, 119, 100, 47, 126, 101, 123,
                      137, 177, 50, 36, 209, 129, 205, 147, 177, 215, 187, 207, 206, 203]
    cheek_right_top = [349, 348, 347, 346, 345, 447, 323,
                       280, 352, 330, 371, 358, 423, 426, 425, 427, 411, 376]
    # dense zones used for convex hull masks
    left_eye = [157,144, 145, 22, 23, 25, 154, 31, 160, 33, 46, 52, 53, 55, 56, 189, 190, 63, 65, 66, 70, 221, 222, 223, 225, 226, 228, 229, 230, 231, 232, 105, 233, 107, 243, 124]
    right_eye = [384, 385, 386, 259, 388, 261, 265, 398, 276, 282, 283, 285, 413, 293, 296, 300, 441, 442, 445, 446, 449, 451, 334, 463, 336, 464, 467, 339, 341, 342, 353, 381, 373, 249, 253, 255]
    mounth = [391, 393, 11, 269, 270, 271, 287, 164, 165, 37, 167, 40, 43, 181, 313, 314, 186, 57, 315, 61, 321, 73, 76, 335, 83, 85, 90, 106]
    # equispaced facial points - mouth and eyes are excluded.
    equispaced_facial_points = [2, 3, 4, 5, 6, 8, 9, 10, 18, 21, 32, 35, 36, 43, 46, 47, 48, 50, 54, \
             58, 67, 68, 69, 71, 92, 93, 101, 103, 104, 108, 109, 116, 117, \
             118, 123, 132, 134, 135, 138, 139, 142, 148, 149, 150, 151, 152, 182, 187, 188, 193, 197, 201, 205, 206, 207, \
             210, 211, 212, 216, 234, 248, 251, 262, 265, 266, 273, 277, 278, 280, \
             284, 288, 297, 299, 322, 323, 330, 332, 333, 337, 338, 345, \
             346, 361, 363, 364, 367, 368, 371, 377, 379, 411, 412, 417, 421, 425, 426, 427, 430, 432, 436]

def get_magic_landmarks():
    """ returns high_priority and mid_priority list of landmarks identification number """
    return [*MagicLandmarks.forehead_center, *MagicLandmarks.cheek_left_bottom, *MagicLandmarks.cheek_right_bottom], [*MagicLandmarks.forehoead_right, *MagicLandmarks.forehead_left, *MagicLandmarks.cheek_left_top, *MagicLandmarks.cheek_right_top]

@njit(parallel=True)
def draw_rects(image, xcenters, ycenters, xsides, ysides, color):
    """
    This method is used to draw N rectangles on a image.
    """
    for idx in prange(len(xcenters)):
        leftx = int(xcenters[idx] - xsides[idx]/2)
        rightx = int(xcenters[idx] + xsides[idx]/2)
        topy = int(ycenters[idx] - ysides[idx]/2)
        bottomy = int(ycenters[idx] + ysides[idx]/2)
        for x in prange(leftx, rightx):
            if topy >= 0 and x >= 0 and x < image.shape[1]:
                image[topy, x, 0] = color[0]
                image[topy, x, 1] = color[1]
                image[topy, x, 2] = color[2]
            if bottomy < image.shape[0] and x >= 0 and x < image.shape[1]:
                image[bottomy, x, 0] = color[0]
                image[bottomy, x, 1] = color[1]
                image[bottomy, x, 2] = color[2]
        for y in prange(topy, bottomy):
            if leftx >= 0 and y >= 0 and y < image.shape[0]:
                image[y, leftx, 0] = color[0]
                image[y, leftx, 1] = color[1]
                image[y, leftx, 2] = color[2]
            if rightx < image.shape[1] and y >= 0 and y < image.shape[0]:
                image[y, rightx, 0] = color[0]
                image[y, rightx, 1] = color[1]
                image[y, rightx, 2] = color[2]
    return image

def sig_windowing(sig, wsize, stride, fps):
    """
    This method is used to divide a RGB signal into overlapping windows.

    Args:
        sig (float32 ndarray): ndarray with shape [num_frames, num_estimators, rgb_channels].
        wsize (float): window size in seconds.
        stride (float): stride between overlapping windows in seconds.
        fps (float): frames per seconds.

    Returns:
        A list of ndarray (float32) with shape [num_estimators, rgb_channels, window_frames],
        an array (float32) of times in seconds (win centers)
    """
    N = sig.shape[0]
    block_idx, timesES = sliding_straded_win_idx(N, wsize, stride, fps)
    block_signals = []
    for e in block_idx:
        st_frame = int(e[0])
        end_frame = int(e[-1])
        wind_signal = np.copy(sig[st_frame: end_frame+1])
        wind_signal = np.swapaxes(wind_signal, 0, 1)
        wind_signal = np.swapaxes(wind_signal, 1, 2)
        block_signals.append(wind_signal)
    return block_signals, timesES

    """
    This method is used to divide a Raw signal into overlapping windows.

    Args:
        sig (float32 ndarray): ndarray of images with shape [num_frames, rows, columns, rgb_channels].
        wsize (float): window size in seconds.
        stride (float): stride between overlapping windows in seconds.
        fps (float): frames per seconds.

    Returns:
        windowed signal as a list of length num_windows of float32 ndarray with shape [num_frames, rows, columns, rgb_channels],
        and a 1D ndarray of times in seconds,where each one is the center of a window.
    """
    N = raw_signal.shape[0]
    block_idx, timesES = sliding_straded_win_idx(N, wsize, stride, fps)
    block_signals = []
    for e in block_idx:
        st_frame = int(e[0])
        end_frame = int(e[-1])
        wind_signal = np.copy(raw_signal[st_frame: end_frame+1])
        # check for zero traces
        sum_wind = np.sum(wind_signal, axis=(1,2))
        zero_idx = np.argwhere(sum_wind == 0).squeeze()
        est_idx = np.ones(wind_signal.shape[0], dtype=bool)
        est_idx[zero_idx] = False
        # append traces
        block_signals.append(wind_signal[est_idx])
    return block_signals, timesES

def sliding_straded_win_idx(N, wsize, stride, fps):
    """
    This method is used to compute the indices for creating an overlapping windows signal.

    Args:
        N (int): length of the signal.
        wsize (float): window size in seconds.
        stride (float): stride between overlapping windows in seconds.
        fps (float): frames per seconds.

    Returns:
        List of ranges, each one contains the indices of a window, and a 1D ndarray of times in seconds, where each one is the center of a window.
    """
    wsize_fr = wsize*fps
    stride_fr = stride*fps
    idx = []
    timesES = []
    num_win = int((N-wsize_fr)/stride_fr)+1
    s = 0
    for i in range(num_win):
        idx.append(np.arange(s, s+wsize_fr))
        s += stride_fr
        timesES.append(wsize/2+stride*i)
    return idx, np.array(timesES, dtype=np.float32)

def get_fps(videoFileName):
    """
    This method returns the fps of a video file name or path.
    """
    vidcap = cv2.VideoCapture(videoFileName)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    vidcap.release()
    return fps

def Welch(bvps, fs):
  _, n = bvps.shape
  if n < 256:
      seglength = n
      overlap = int(0.6*n)  # fixed overlapping
  else:
      seglength = 256
      overlap = 200
  # -- periodogram by Welch

  if np.isnan(bvps).any():
    print('OK bvp')
  F, P = welch(bvps, nperseg=seglength, noverlap=overlap, fs=fs, nfft=2048)
  F = F.astype(np.float32)
  P = P.astype(np.float32)
  # -- freq subband (0.65 Hz - 4.0 Hz)
  band = np.argwhere((F > 0.65) & (F < 4.0)).flatten()
  Pfreqs = 60*F[band]
  Power = P[:, band]
  return Pfreqs, Power

def extract_frames_yield(videoFileName):
    """
    This method yield the frames of a video file name or path.
    """
    vidcap = cv2.VideoCapture(videoFileName)
    success, image = vidcap.read()
    while success:
        yield image
        success, image = vidcap.read()
    vidcap.release()

def med_mad(x):
  MED = np.median(x)
  MAD = median_abs_deviation(x)
  return MED, MAD

def adjust_BMPs(bpmES, bpmES0, bpmES1, wsize, thr=10):
  """
    adjust the final estimate by evaluating whether to replace the chosen value 
    with the discarded one based on the distance between the current value and the median 
    Args:
        bpmES: the estimates chosen.
        bpmES0: the estimates given by the first cluster.
        bpmES1: the estimates given by the second cluster.
        wsize: thr size of the windows used (in seconds)
        thr: a multiplier factor for the MAD.
    Returns:
        bpmES: possibly redefined bpm estimates.
    """
  T = thr
  N = len(bpmES)
  bpmES = np.array(bpmES)
  bpmES0 = np.array(bpmES0)
  bpmES1 = np.array(bpmES1)
  for i in range(N):  
    L = max(0,i-wsize)
    R = min(N,i+wsize)
    MED, MAD = med_mad(bpmES[L:R])
    diff = np.abs(bpmES[i]-MED)      # all diffs with MED    
    if diff > T+MAD:
      if abs(bpmES[i]-bpmES0[i]) < 10e-9 and abs(MED-bpmES1[i]) < T+MAD:
        bpmES[i] = bpmES1[i]
      elif abs(bpmES[i]-bpmES1[i]) < 10e-9 and abs(MED-bpmES0[i]) < T+MAD:
        bpmES[i] = bpmES0[i]
      else:
        bpmES[i] = MED
  return bpmES
    