# BSD 3-Clause “New” or “Revised” License
#
# Copyright (c) 2016 Idiap Research Institute, http://www.idiap.ch/
# Written by Guillaume Heusch <guillaume.heusch@idiap.ch>
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#
# Original authors:
# Guillaume HEUSCH <guillaume.heusch@idiap.ch> 29/07/2015
#
# Contributors:
# Michele Maione <mikymaione@hotmail.it> 07/03/2019

import numpy


class SkinColorFilter():
    """
    This class implements a number of functions to perform skin color filtering.
    It is based on the work published in "Adaptive skin segmentation via feature-based face detection",
    M.J. Taylor and T. Morris, Proc SPIE Photonics Europe, 2014 [taylor-spie-2014]_
    Attributes
    ----------
    mean: numpy.ndarray | dim: 2
        the mean skin color
    covariance: numpy.ndarray | dim: 2x2
        the covariance matrix of the skin color
    covariance_inverse: numpy.ndarray | dim: 2x2
        the inverse covariance matrix of the skin color
    circular_mask: numpy.ndarray
        mask of the size of the image, defining a circular region in the center
    luma_mask: numpy.ndarray
        mask of the size of the image, defining valid luma values
    """

    def __init__(self):
        self.mean = numpy.array([0.0, 0.0])
        self.covariance = numpy.zeros((2, 2), 'float64')
        self.covariance_inverse = numpy.zeros((2, 2), 'float64')

    def __generate_circular_mask(self, image, radius_ratio=0.4):

        w = image.shape[0]
        h = image.shape[1]
        radius = radius_ratio * h

        x_center = h / 2
        y_center = w / 2

        center = [int(x_center), int(y_center)]
    
        Y, X = numpy.ogrid[:w, :h]
        dist_from_center = numpy.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius

        self.circular_mask = mask

    '''def __generate_circular_mask(self, image, radius_ratio=0.4):
        """
        This function will generate a circular mask to be applied to the image.
        The mask will be true for the pixels contained in a circle centered in the image center, and with radius equals to radius_ratio * the image's height.
        Parameters
        ----------
        image: numpy.ndarray
            The face image.
        radius_ratio: float:
            The ratio of the image's height to define the radius of the circular region. Defaults to 0.4.
        """
        w = image.shape[0]
        h = image.shape[1]

        x_center = w / 2
        y_center = h / 2

        # arrays with the image coordinates
        X = numpy.zeros((w, h))

        import code
        code.interact(local=locals())

        X[:] = range(0, w)

        Y = numpy.zeros((h, w))
        Y[:] = range(0, h)
        Y = numpy.transpose(Y)

        # translate s.t. the center is the origin
        X -= x_center
        Y -= y_center

        # condition to be inside of a circle: x^2 + y^2 < r^2
        radius = radius_ratio * h

        # x ^ 2 + y ^ 2 < r ^ 2
        cm = (X ** 2 + Y ** 2) < (radius ** 2)  # dim : w x h
        self.circular_mask = cm'''

    def __remove_luma(self, image):
        """
        This function remove pixels with extreme luma values.
        Some pixels are considered as non-skin if their intensity is either too high or too low.
        The luma value for all pixels inside a provided circular mask is calculated. Pixels for which the luma value deviates more than 1.5 * standard deviation are pruned.
        Parameters
        ----------
        image: numpy.ndarray
            The face image.
        """

        # compute the mean and std of luma values on non-masked pixels only
        R = 0.299 * image[self.circular_mask, 0]
        G = 0.587 * image[self.circular_mask, 1]
        B = 0.114 * image[self.circular_mask, 2]

        luma = R + G + B

        m = numpy.mean(luma)
        s = numpy.std(luma)

        # apply the filtering to the whole image to get the luma mask
        R = 0.299 * image[:, :, 0]
        G = 0.587 * image[:, :, 1]
        B = 0.114 * image[:, :, 2]

        luma = R + G + B

        # dim : image.x x image.y
        lm = numpy.logical_and((luma > (m - 1.5 * s)), (luma < (m + 1.5 * s)))
        self.luma_mask = lm

    def __RG_Mask(self, image, dtype=None):
        # dim: image.x x image.y
        channel_sum = image[:, :, 0].astype('float64') + image[:, :, 1] + image[:, :, 2]

        # dim: image.x x image.y
        nonzero_mask = numpy.logical_or(numpy.logical_or(image[:, :, 0] > 0, image[:, :, 1] > 0), image[:, :, 2] > 0)

        # dim: image.x x image.y
        R = numpy.zeros((image.shape[0], image.shape[1]), dtype)
        R[nonzero_mask] = image[nonzero_mask, 0] / channel_sum[nonzero_mask]

        # dim: image.x x image.y
        G = numpy.zeros((image.shape[0], image.shape[1]), dtype)
        G[nonzero_mask] = image[nonzero_mask, 1] / channel_sum[nonzero_mask]

        return R, G

    def estimate_gaussian_parameters(self, image):
        """
        This function estimates the parameter of the skin color distribution.
        The mean and covariance matrix of the skin pixels in the normalised rg colorspace are computed.
        Note that only the pixels for which both the circular and the luma mask is 'True' are considered.
        Parameters
        ----------
        image: numpy.ndarray
            The face image.
        """
        self.__generate_circular_mask(image)
        self.__remove_luma(image)

        # dim: image.x x image.y
        mask = numpy.logical_and(self.luma_mask, self.circular_mask)

        # get the mean
        # R dim: image.x x image.y
        # G dim: image.x x image.y
        R, G = self.__RG_Mask(image)

        # dim: 2
        self.mean = numpy.array([numpy.mean(R[mask]), numpy.mean(G[mask])])

        # get the covariance
        R_minus_mean = R[mask] - self.mean[0]
        G_minus_mean = G[mask] - self.mean[1]

        samples = numpy.vstack((R_minus_mean, G_minus_mean))
        samples = samples.T

        cov = sum([numpy.outer(s, s) for s in samples])  # dim: 2x2

        self.covariance = cov / float(samples.shape[0] - 1)

        # store the inverse covariance matrix (no need to recompute)
        if numpy.linalg.det(self.covariance) != 0:
            self.covariance_inverse = numpy.linalg.inv(self.covariance)
        else:
            self.covariance_inverse = numpy.zeros_like(self.covariance)

    def get_skin_mask(self, image, threshold=0.5):
        """
        This function computes the probability of skin-color for each pixel in the image.
        Parameters
        ----------
        image: numpy.ndarray
            The face image.
        threshold: float: 0->1
            The threshold on the skin color probability. Defaults to 0.5
        Returns
        -------
        skin_mask: numpy.ndarray
            The mask where skin color pixels are labeled as True.
        """

        # get the image in rg colorspace
        R, G = self.__RG_Mask(image, 'float64')

        # compute the skin probability map
        R_minus_mean = R - self.mean[0]
        G_minus_mean = G - self.mean[1]

        n = R.shape[0] * R.shape[1]
        V = numpy.dstack((R_minus_mean, G_minus_mean))  # dim: image.x x image.y
        V = V.reshape((n, 2))  # dim: nx2

        probs = [numpy.dot(k, numpy.dot(self.covariance_inverse, k)) for k in V]
        probs = numpy.array(probs).reshape(R.shape)  # dim: image.x x image.y

        skin_map = numpy.exp(-0.5 * probs)  # dim: image.x x image.y

        return skin_map > threshold