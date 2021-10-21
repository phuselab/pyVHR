from numba import prange, njit
import numpy as np
import cv2

def get_magic_landmarks():
    """ returns high_priority and mid_priority list of landmarks identification number """
    return [*MagicLandmarks.forehead_center, *MagicLandmarks.cheek_left_bottom, *MagicLandmarks.cheek_right_bottom], [*MagicLandmarks.forehoead_right, *MagicLandmarks.forehead_left, *MagicLandmarks.cheek_left_top, *MagicLandmarks.cheek_right_top]


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
        windowed signal as a list of length num_windows of float32 ndarray with shape [num_estimators, rgb_channels, window_frames],
        and a 1D ndarray of times in seconds,where each one is the center of a window.
    """
    N = sig.shape[0]
    block_idx, timesES = sliding_straded_win_offline(N, wsize, stride, fps)
    block_signals = []
    for e in block_idx:
        st_frame = int(e[0])
        end_frame = int(e[-1])
        wind_signal = np.copy(sig[st_frame: end_frame+1])
        wind_signal = np.swapaxes(wind_signal, 0, 1)
        wind_signal = np.swapaxes(wind_signal, 1, 2)
        block_signals.append(wind_signal)
    return block_signals, timesES

def raw_windowing(raw_signal, wsize, stride, fps):
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
    block_idx, timesES = sliding_straded_win_offline(N, wsize, stride, fps)
    block_signals = []
    for e in block_idx:
        st_frame = int(e[0])
        end_frame = int(e[-1])
        wind_signal = np.copy(raw_signal[st_frame: end_frame+1])
        block_signals.append(wind_signal)
    return block_signals, timesES

def sliding_straded_win_offline(N, wsize, stride, fps):
    """
    This method is used to compute all the info for creating an overlapping windows signal.

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