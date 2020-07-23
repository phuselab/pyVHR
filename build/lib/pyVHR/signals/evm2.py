#!/usr/bin/env python

"""
Eulerian Video Magnification (EVM) Demo
"""

import time
import sys

import cv2
import numpy as np
import scipy
import skimage

def gaussian(image, numlevels):
	"""Constructs gaussian pyramid

	Arguments:
		image : Input image (monochrome or color)
		numlevels : Number of levels to compute

	Return:
		List of progressively smaller (i.e. lower frequency) images
	"""

	pyramid = [ image ]
	for level in range(numlevels):
		image = cv2.pyrDown(image)
		pyramid.append(image)

	return pyramid

def temporal_bandpass_filter(data, freq_min, freq_max, fps, axis=0):
	"""Applies ideal band-pass filter to a given video
	Arguments:
		data : video to be filtered (as a 4-d numpy array (time, height,
		        width, channels))
		freq_min : lower cut-off frequency of band-pass filter
		freq_max : upper cut-off frequency of band-pass filter
		fps :
	Return:
		Temporally filtered video as 4-d array
	"""

	# perform FFT on each frame
	fft = scipy.fftpack.fft(data, axis=axis)
	# sampling frequencies, where the step d is 1/samplingRate
	frequencies = scipy.fftpack.fftfreq(data.shape[0], d=1.0 / fps)
	# find the indices of low cut-off frequency
	bound_low = (np.abs(frequencies - freq_min)).argmin()
	# find the indices of high cut-off frequency
	bound_high = (np.abs(frequencies - freq_max)).argmin()
	# band pass filtering
	fft[:bound_low] = 0
	fft[-bound_low:] = 0
	fft[bound_high:-bound_high] = 0
	# perform inverse FFT
	return np.real(scipy.fftpack.ifft(fft, axis=0))

class EVM():
	"""Eulerian Video Magnification"""

	def __init__(self, frames, fps):
		"""Constructor"""
		self.fps = fps
		self.frames = frames
		self.frameCount = len(frames)
		self.frameHeight = int(frames[0].shape[0])
		self.frameWidth = int(frames[0].shape[1])
		self.numChannels = 3
		# allocate memory for input frames
		self.in_frames = frames
		self.out_frames = frames

	def process(self, numlevels=4, alpha=50., chromAttenuation=1., lowcut=0.5, highcut=1.5):
		"""Process video

		Arguments:
			numlevels : Number of pyramid levels to compute
		"""
		# compute pyramid on first frame
		pyramid = gaussian(self.in_frames[0], numlevels)
		height, width, _ = pyramid[-1].shape

		# allocate memory for downsampled frames
		self.ds_frames = np.ndarray(shape=(self.frameCount, \
		                                   height, \
		                                   width, \
		                                   self.numChannels), \
		                            dtype=np.float32)
		self.ds_frames[0] = pyramid[-1]

		for frameNumber in range(1, self.frameCount):

			# spatial decomposition (specify laplacian or gaussian)
			pyramid = gaussian(self.in_frames[frameNumber], numlevels)

			# store downsampled frame into memory
			self.ds_frames[frameNumber] = pyramid[-1]

		#print ('filtering...')
		output = temporal_bandpass_filter(self.ds_frames, lowcut, highcut, self.fps)

		#print ('amplifying...')
		output[:,:,:,0] *= alpha
		output[:,:,:,1] *= (alpha * chromAttenuation)
		output[:,:,:,2] *= (alpha * chromAttenuation)

		for i in range(self.frameCount):

			orig = self.in_frames[i]

			filt = output[i].astype(np.float32)

			# enlarge to match size of original frame (keep as 32-bit float)
			filt = cv2.resize(filt, (self.frameWidth, self.frameHeight), interpolation=cv2.INTER_CUBIC)

			filt = filt + orig

			filt = skimage.color.yiq2rgb(filt)

			#filt[filt > 1] = 1
			#filt[filt < 0] = 0

			self.out_frames[i] = filt

		return self.out_frames

def main(frames, fps, alpha, numlevels):
	evm = EVM(frames, fps)
	filt = evm.process(alpha=alpha, numlevels=numlevels)

	return filt
