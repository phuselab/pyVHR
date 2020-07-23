"""
A utils module used in the actual evm module performing such tasks as
pyramid construction, video io and filter application

functions were originally written by flyingzhao but adapted for this module
"""

import cv2
import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack

def build_gaussian_pyramid(src, levels=3):
	"""
	Function: build_gaussian_pyramid
	--------------------------------
		Builds a gaussian pyramid

	Args:
	-----
		src: the input image
		levels: the number levels in the gaussian pyramid

	Returns:
	--------
		A gaussian pyramid
	"""
	s=src.copy()
	pyramid=[s]
	for i in range(levels):
		s=cv2.pyrDown(s)
		pyramid.append(s)
	return pyramid


def gaussian_video(video, levels=3):
	"""
	Function: gaussian_video
	------------------------
		generates a gaussian pyramid for each frame in a video

	Args:
	-----
		video: the input video array
		levels: the number of levels in the gaussian pyramid

	Returns:
	--------
		the gaussian video
	"""
	n = video.shape[0]
	for i in range(0, n):
		pyr = build_gaussian_pyramid(video[i], levels=levels)
		gaussian_frame=pyr[-1]
		if i==0:
			vid_data = np.zeros((n, *gaussian_frame.shape))
		vid_data[i] = gaussian_frame
	return vid_data


def reconstruct_video_g(amp_video, original_video, levels=3):
	"""
	Function: reconstruct_video_g
	-----------------------------
		reconstructs a video from a gaussian pyramid and the original

	Args:
	-----
		amp_video: the amplified gaussian video
		original_video: the original video
		levels: the levels in the gaussian video

	Returns:
	--------
		the reconstructed video
	"""
	final_video = np.zeros(original_video.shape)
	for i in range(0, amp_video.shape[0]):
		img = amp_video[i]
		for x in range(levels):
			img = cv2.pyrUp(img)
		img = img + original_video[i]
		final_video[i] = img
	return final_video


def build_laplacian_pyramid(src,levels=3):
	"""
	Function: build_laplacian_pyramid
	---------------------------------
		Builds a Laplacian Pyramid

	Args:
	-----
		src: the input image
		levels: the number levels in the laplacian pyramid

	Returns:
	--------
		A Laplacian pyramid
	"""
	gaussianPyramid = build_gaussian_pyramid(src, levels)
	pyramid=[]
	for i in range(levels,0,-1):
		GE=cv2.pyrUp(gaussianPyramid[i])
		L=cv2.subtract(gaussianPyramid[i-1],GE)
		pyramid.append(L)
	return pyramid


def laplacian_video(video, levels=3):
	"""
	Function: laplacian_video
	-------------------------
		generates a laplaican pyramid for each frame in a video

	Args:
	-----
		video: the input video array
		levels: the number of levels for each laplacian pyramid

	Returns:
	--------
		The laplacian video
	"""
	tensor_list=[]
	n = video.shape[0]
	for i in range(0, n):
		frame=video[i]
		pyr = build_laplacian_pyramid(frame,levels=levels)
		if i==0:
			for k in range(levels):
				tensor_list.append(np.zeros((n, *pyr[k].shape)))
		for n in range(levels):
			tensor_list[n][i] = pyr[n]
	return tensor_list


def reconstruct_video_l(lap_pyr, levels=3):
	"""
	Function: reconstruct_video_l
	-----------------------------
		reconstructs a video from a laplacian pyramid and the original

	Args:
	-----
		lap_pyr: the amplified laplacian pyramid
		levels: the levels in the laplacian video

	Returns:
	--------
		the reconstructed video
	"""
	final = np.zeros(lap_pyr[-1].shape)
	for i in range(lap_pyr[0].shape[0]):
		up = lap_pyr[0][i]
		for n in range(levels-1):
			up = cv2.pyrUp(up) + lap_pyr[n + 1][i]
		final[i] = up
	return final


def save_video(video, filename='out.avi'):
	"""
	Function: save_video
	--------------------
		saves a video to a file

	Args:
	-----
		video: the numpy array representing the video
		filename: the name of the output file

	Returns:
		None
	"""
	fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
	n, h, w, _ = video.shape
	writer = cv2.VideoWriter(filename, fourcc, 30, (w, h), 1)
	for i in range(0, n):
		writer.write(cv2.convertScaleAbs(video[i]))
	writer.release()


def load_video(video_filename):
	"""
	Function: load_video
	--------------------
		Loads a video from a file

	Args:
	-----
		video_filename: the name of the video file

	Returns:
	--------
		a numpy array with shape (num_frames, height, width, channels)
		the frame rate of the video
	"""
	cap = cv2.VideoCapture(video_filename)

	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = int(cap.get(cv2.CAP_PROP_FPS))

	video = np.zeros((frame_count, height, width, 3), dtype='float')
	x = 0
	while cap.isOpened():
		ret, frame = cap.read()
		if ret is True:
			video[x] = frame
			x += 1
		else:
			break
	return video, fps


def temporal_ideal_filter(arr, low, high, fps, axis=0):
	"""
	Function: temporal_ideal_filter
	-------------------------------
		Applies a temporal ideal filter to a numpy array

	Args:
	-----
		arr: a numpy array with shape (N, H, W, C)
			N: number of frames
			H: height
			W: width
			C: channels
		low: the low frequency bound
		high: the high frequency bound
		fps: the video frame rate
		axis: the axis of video, should always be 0

	Returns:
	--------
		the array with the filter applied
	"""
	fft = fftpack.fft(arr, axis=axis)
	frequencies = fftpack.fftfreq(arr.shape[0], d=1.0 / fps)
	bound_low = (np.abs(frequencies - low)).argmin()
	bound_high = (np.abs(frequencies - high)).argmin()
	fft[:bound_low] = 0
	fft[bound_high:-bound_high] = 0
	fft[-bound_low:] = 0
	iff=fftpack.ifft(fft, axis=axis)
	return np.abs(iff)


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	"""
	Function: butter_bandpass_filter
	--------------------------------
		applies a buttersworth bandpass filter

	Args:
	-----
		data: the input data
		lowcut: the low cut value
		highcut: the high cut value
		fs: the frame rate in frames per second
		order: the order for butter

	Returns:
	--------
		the result of the buttersworth bandpass filter
	"""
	omega = 0.5 * fs
	low = lowcut / omega
	high = highcut / omega
	b, a = signal.butter(order, [low, high], btype='band')
	y = signal.lfilter(b, a, data, axis=0)
	return y
