import numpy as np
from scipy import signal
import sys
import cv2
from pyVHR.utils.HDI import hdi, hdi2

class SkinDetect():

    def __init__(self, strength=0.2):
        self.description = 'Skin Detection Module'
        self.strength = strength
        self.stats_computed = False

    def compute_stats(self, face):

        assert (self.strength > 0 and self.strength < 1), "'strength' parameter must have values in [0,1]"

        faceColor = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)
        h = faceColor[:,:,0].reshape(-1,1)
        s = faceColor[:,:,1].reshape(-1,1)
        v = faceColor[:,:,2].reshape(-1,1)
       
        alpha = self.strength    #the highest, the stronger the masking  

        hpd_h, x_h, y_h, modes_h = hdi2(np.squeeze(h), alpha=alpha)
        min_s, max_s = hdi(np.squeeze(s), alpha=alpha)
        min_v, max_v = hdi(np.squeeze(v), alpha=alpha)

        if len(hpd_h) > 1:

            self.multiple_modes = True

            if len(hpd_h) > 2:
                print('WARNING!! Found more than 2 HDIs in Hue Channel empirical Distribution... Considering only 2')
                from scipy.spatial.distance import pdist, squareform
                m = np.array(modes_h).reshape(-1,1)
                d = squareform(pdist(m))
                maxij = np.where(d==d.max())[0]
                i = maxij[0]
                j = maxij[1]
            else:
                i = 0
                j = 1

            min_h1 = hpd_h[i][0]
            max_h1 = hpd_h[i][1]
            min_h2 = hpd_h[j][0]
            max_h2 = hpd_h[j][1]
            
            self.lower1 = np.array([min_h1, min_s, min_v], dtype = "uint8")
            self.upper1 = np.array([max_h1, max_s, max_v], dtype = "uint8")
            self.lower2 = np.array([min_h2, min_s, min_v], dtype = "uint8")
            self.upper2 = np.array([max_h2, max_s, max_v], dtype = "uint8")

        elif len(hpd_h) == 1:

            self.multiple_modes = False
            
            min_h = hpd_h[0][0]
            max_h = hpd_h[0][1]
            
            self.lower = np.array([min_h, min_s, min_v], dtype = "uint8")
            self.upper = np.array([max_h, max_s, max_v], dtype = "uint8")

        self.stats_computed = True


    def get_skin(self, face, filt_kern_size=7, verbose=False, plot=False):

        if not self.stats_computed:
            raise ValueError("ERROR! You must compute stats at least one time")

        faceColor = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)

        if self.multiple_modes:
            if verbose:
                print('\nLower1: ' + str(self.lower1))
                print('Upper1: ' + str(self.upper1))
                print('\nLower2: ' + str(self.lower2))
                print('Upper2: ' + str(self.upper2) + '\n')

            skinMask1 = cv2.inRange(faceColor, self.lower1, self.upper1)
            skinMask2 = cv2.inRange(faceColor, self.lower2, self.upper2)
            skinMask = np.logical_or(skinMask1, skinMask2).astype(np.uint8)*255

        else:

            if verbose:
                print('\nLower: ' + str(lower))
                print('Upper: ' + str(upper) + '\n')

            skinMask = cv2.inRange(faceColor, self.lower, self.upper)

        if filt_kern_size > 0:
            skinMask = signal.medfilt2d(skinMask, kernel_size=filt_kern_size)
        skinFace = cv2.bitwise_and(face, face, mask=skinMask)

        if plot:
            
            h = faceColor[:,:,0].reshape(-1,1)
            s = faceColor[:,:,1].reshape(-1,1)
            v = faceColor[:,:,2].reshape(-1,1)

            import matplotlib.pyplot as plt
            plt.figure()              
            plt.subplot(2,2,1)               
            plt.hist(h, 20)
            plt.title('Hue')
            plt.subplot(2,2,2)
            plt.hist(s, 20)
            plt.title('Saturation')
            plt.subplot(2,2,3)
            plt.hist(v, 20)
            plt.title('Value')
            plt.subplot(2,2,4)
            plt.imshow(skinFace)
            plt.title('Masked Face')
            plt.show()

        return skinFace
