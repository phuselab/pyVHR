from numba import cuda, njit, prange
import cupy
import math
import numpy as np
import torchvision.transforms as transforms
import torch
from numba import prange, njit, cuda
from pyVHR.resources.faceparsing.model import BiSeNet
import os
import pyVHR
from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw
import requests

"""
This module defines classes or methods used for skin extraction.
"""

### functions and parameters ###

class SkinProcessingParams():
    """
        This class contains usefull parameters used by this module.

        RGB_LOW_TH (numpy.int32): RGB low-threshold value.

        RGB_HIGH_TH (numpy.int32): RGB high-threshold value.
    """
    RGB_LOW_TH = np.int32(55)
    RGB_HIGH_TH = np.int32(200)


def bbox2_CPU(img):
    """
    Args:
        img (ndarray): ndarray with shape [rows, columns, rgb_channels].

    Returns: 
        Four cropping coordinates (row, row, column, column) for removing black borders (RGB [O,O,O]) from img.
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    nzrows = np.nonzero(rows)
    nzcols = np.nonzero(cols)
    if nzrows[0].size == 0 or nzcols[0].size == 0:
        return -1, -1, -1, -1
    rmin, rmax = np.nonzero(rows)[0][[0, -1]]
    cmin, cmax = np.nonzero(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def bbox2_GPU(img):
    """
    Args:
        img (cupy.ndarray): cupy.ndarray with shape [rows, columns, rgb_channels].

    Returns: 
        Four cropping coordinates (row, row, column, column) for removing black borders (RGB [O,O,O]) from img. 
        The returned variables are on GPU.
    """
    rows = cupy.any(img, axis=1)
    cols = cupy.any(img, axis=0)
    nzrows = cupy.nonzero(rows)
    nzcols = cupy.nonzero(cols)
    if nzrows[0].size == 0 or nzcols[0].size == 0:
        return -1, -1, -1, -1
    rmin, rmax = cupy.nonzero(rows)[0][[0, -1]]
    cmin, cmax = cupy.nonzero(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


### SKIN EXTRACTION CLASSES ###

class SkinExtractionFaceParsing():
    """
        This class performs skin extraction on CPU/GPU using Face Parsing.
        https://github.com/zllrunning/face-parsing.PyTorch
    """
    def __init__(self, device='CPU'):
        """
        Args:
            device (str): This class can execute code on 'CPU' or 'GPU'.
        """
        self.device = device
        n_classes = 19
        self.net = BiSeNet(n_classes=n_classes)
        if self.device == 'GPU':
            self.net.cuda()
            self.kernel_cuda_skin_copy_and_filter = kernel_cuda_skin_copy_and_filter()
        save_pth = os.path.dirname(
            pyVHR.resources.faceparsing.model.__file__) + "/79999_iter.pth"
        if not os.path.isfile(save_pth):
            url = "https://github.com/phuselab/pyVHR/blob/main/pyVHR/resources/faceparsing/79999_iter.pth"
            print('Downloading faceparsing model...')
            r = requests.get(url, allow_redirects=True)
            open(save_pth, 'wb').write(r.content)    
        self.net.load_state_dict(torch.load(save_pth))
        self.net.eval()
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def extract_skin(self, image, ldmks):
        """
        This method extract the skin from an image using Face Parsing. 
        Landmarks (ldmks) are used to create a facial bounding box for cropping the face; this way
        the network used in Face Parsing is more accurate. 

        Args:
            image (uint8 ndarray): ndarray with shape [rows, columns, rgb_channels].
            ldmks (float32 ndarray): ndarray with shape [num_landmarks, xy_coordinates].

        Returns:
            Cropped skin-image and non-cropped skin-image; both are uint8 ndarray with shape [rows, columns, rgb_channels].
        """
        # crop with bounding box of ldmks; the network works better if the bounding box is bigger
        aviable_ldmks = ldmks[ldmks[:,0] >= 0][:,:2]  
        min_y, min_x = np.min(aviable_ldmks, axis=0)
        max_y, max_x = np.max(aviable_ldmks, axis=0)
        min_y *= 0.90
        min_x *= 0.90
        max_y = max_y * 1.10 if max_y * 1.10 < image.shape[0] else image.shape[0]
        max_x = max_x * 1.10 if max_x * 1.10 < image.shape[1] else image.shape[1]
        cropped_image = np.copy(image[int(min_y):int(max_y),int(min_x):int(max_x) ,:])
        nda_im = np.array(cropped_image)
        # prepare the image for the bisenet network
        cropped_image = self.to_tensor(cropped_image)
        cropped_image = torch.unsqueeze(cropped_image, 0)
        cropped_skin_img = self.extraction(cropped_image, nda_im)
        # recreate full image using cropped_skin_img
        full_skin_image = np.zeros_like(image)
        full_skin_image[int(min_y):int(max_y),int(min_x):int(max_x) ,:] = cropped_skin_img
        return cropped_skin_img, full_skin_image

    def extraction(self, im, nda_im):
        """
        This method performs skin extraction using Face Parsing.

        Args:
            im (torch.Tensor): torch.Tensor with size [rows, columns, rgb_channels] 
            nda_im (uint8 ndarray): ndarray with shape [rows, columns, rgb_channels].

        Returns:
            skin-image as uint8 ndarray with shape [rows, columns, rgb_channels].
        """
        with torch.no_grad():
            if self.device == 'CPU':
                ### bisenet skin detection ###
                out = self.net(im)[0]
                ### gpu cuda skin copy ###
                parsing = out.squeeze(0).argmax(0).numpy()
                parsing = parsing.astype(np.int32)
                nda_im = nda_im.astype(np.uint8)
                # kernel skin copy
                return kernel_skin_copy_and_filter(nda_im, parsing, np.int32(SkinProcessingParams.RGB_LOW_TH), np.int32(SkinProcessingParams.RGB_HIGH_TH))
            else:
                ### bisenet skin detection ###
                im = im.cuda()
                out = self.net(im)[0]
                ### gpu cuda skin copy ###
                dev_parsing = out.squeeze(0).argmax(0).type(torch.int32)
                dev_nda_im = cuda.to_device(nda_im)
                dev_new_im = cuda.to_device(np.zeros_like(nda_im))
                low_high_f = cuda.to_device(
                    np.array([SkinProcessingParams.RGB_LOW_TH, SkinProcessingParams.RGB_HIGH_TH], dtype=np.int32))
                # define number of blocks and threads per block
                threadsperblock = (16, 16)
                blockspergrid_x = math.ceil(nda_im.shape[0] / threadsperblock[0])
                blockspergrid_y = math.ceil(nda_im.shape[1] / threadsperblock[1])
                blockspergrid = (blockspergrid_x, blockspergrid_y)
                # kernel invoke
                self.kernel_cuda_skin_copy_and_filter[blockspergrid, threadsperblock](
                    dev_nda_im, dev_parsing, dev_new_im, low_high_f)
                # copy to CPU result
                newimg = dev_new_im.copy_to_host()
                # free memory
                im = None
                out = None
                dev_nda_im = None
                dev_parsing = None
                dev_new_im = None
                return newimg

# SkinExtractionFaceParsing class kernels #
def kernel_cuda_skin_copy_and_filter():
    """
    Return a Numba cuda.jit kernel defined as:

    @cuda.jit('void(uint8[:,:,:], int32[:,:], uint8[:,:,:], int32[:])')
    def __kernel_cuda_skin_copy_and_filter(orig, pars, new, low_high_filter):
        ''
        This method removes pixels from the image 'orig' that are not skin, or 
        that are outside the RGB range [low_high_filter[0], low_high_filter[1]] (extremes are included).
        ''
  
    This method is important for users who do not use a GPU, beacause they can't compile @cuda.jit.
    """

    @cuda.jit('void(uint8[:,:,:], int32[:,:], uint8[:,:,:], int32[:])')
    def __kernel_cuda_skin_copy_and_filter(orig, pars, new, low_high_filter):
        """
        This method removes pixels from the image 'orig' that are not skin, or 
        that are outside the RGB range [low_high_filter[0], low_high_filter[1]] (extremes are included).
        """
        x, y = cuda.grid(2)
        if x < orig.shape[0] and y < orig.shape[1]:
            # skin class = 1, nose = 10
            if pars[x, y] == 1 or pars[x, y] == 10:
                if not ((orig[x, y, 0] <= low_high_filter[0] and orig[x, y, 1] <= low_high_filter[0] and orig[x, y, 2] <= low_high_filter[0]) or
                        (orig[x, y, 0] >= low_high_filter[1] and orig[x, y, 1] >= low_high_filter[1] and orig[x, y, 2] >= low_high_filter[1])):
                    new[x, y, 0] = orig[x, y, 0]
                    new[x, y, 1] = orig[x, y, 1]
                    new[x, y, 2] = orig[x, y, 2]

    return __kernel_cuda_skin_copy_and_filter


@njit('uint8[:,:,:](uint8[:,:,:], int32[:,:], int32, int32)', parallel=True, nogil=True)
def kernel_skin_copy_and_filter(orig, pars, RGB_LOW_TH, RGB_HIGH_TH):
    """
    This method removes pixels from the image 'orig' that are not skin, or 
    that are outside the RGB range [RGB_LOW_TH, RGB_HIGH_TH] (extremes are included).
    """
    new = np.zeros_like(orig)
    for x in prange(orig.shape[0]):
        for y in prange(orig.shape[1]):
            # skin class = 1, nose = 10
            if pars[x, y] == 1 or pars[x, y] == 10:
                if not ((orig[x, y, 0] <= RGB_LOW_TH and orig[x, y, 1] <= RGB_LOW_TH and orig[x, y, 2] <= RGB_LOW_TH) or
                        (orig[x, y, 0] >= RGB_HIGH_TH and orig[x, y, 1] >= RGB_HIGH_TH and orig[x, y, 2] >= RGB_HIGH_TH)):
                    new[x, y, 0] = orig[x, y, 0]
                    new[x, y, 1] = orig[x, y, 1]
                    new[x, y, 2] = orig[x, y, 2]
    return new

class SkinExtractionConvexHull:
    """
        This class performs skin extraction on CPU/GPU using a Convex Hull segmentation obtained from facial landmarks.
    """
    def __init__(self,device='CPU'):
        """
        Args:
            device (str): This class can execute code on 'CPU' or 'GPU'.
        """
        self.device = device
    
    def extract_skin(self,image, ldmks):
        """
        This method extract the skin from an image using Convex Hull segmentation.

        Args:
            image (uint8 ndarray): ndarray with shape [rows, columns, rgb_channels].
            ldmks (float32 ndarray): landmarks used to create the Convex Hull; ldmks is a ndarray with shape [num_landmarks, xy_coordinates].

        Returns:
            Cropped skin-image and non-cropped skin-image; both are uint8 ndarray with shape [rows, columns, rgb_channels].
        """
        from pyVHR.extraction.sig_processing import MagicLandmarks
        aviable_ldmks = ldmks[ldmks[:,0] >= 0][:,:2]        
        # face_mask convex hull 
        hull = ConvexHull(aviable_ldmks)
        verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
        img = Image.new('L', image.shape[:2], 0)
        ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
        mask = np.array(img)
        mask = np.expand_dims(mask,axis=0).T

        # left eye convex hull
        left_eye_ldmks = ldmks[MagicLandmarks.left_eye]
        aviable_ldmks = left_eye_ldmks[left_eye_ldmks[:,0] >= 0][:,:2]
        if len(aviable_ldmks) > 3:
            hull = ConvexHull(aviable_ldmks)
            verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
            img = Image.new('L', image.shape[:2], 0)
            ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
            left_eye_mask = np.array(img)
            left_eye_mask = np.expand_dims(left_eye_mask,axis=0).T
        else:
            left_eye_mask = np.ones((image.shape[0], image.shape[1],1),dtype=np.uint8)

        # right eye convex hull
        right_eye_ldmks = ldmks[MagicLandmarks.right_eye]
        aviable_ldmks = right_eye_ldmks[right_eye_ldmks[:,0] >= 0][:,:2]
        if len(aviable_ldmks) > 3:
            hull = ConvexHull(aviable_ldmks)
            verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
            img = Image.new('L', image.shape[:2], 0)
            ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
            right_eye_mask = np.array(img)
            right_eye_mask = np.expand_dims(right_eye_mask,axis=0).T
        else:
            right_eye_mask = np.ones((image.shape[0], image.shape[1],1),dtype=np.uint8)

        # mounth convex hull
        mounth_ldmks = ldmks[MagicLandmarks.mounth]
        aviable_ldmks = mounth_ldmks[mounth_ldmks[:,0] >= 0][:,:2]
        if len(aviable_ldmks) > 3:
            hull = ConvexHull(aviable_ldmks)
            verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
            img = Image.new('L', image.shape[:2], 0)
            ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
            mounth_mask = np.array(img)
            mounth_mask = np.expand_dims(mounth_mask,axis=0).T
        else:
            mounth_mask = np.ones((image.shape[0], image.shape[1],1),dtype=np.uint8)

        # apply masks and crop 
        if self.device == 'GPU':
            image = cupy.asarray(image)
            mask = cupy.asarray(mask)
            left_eye_mask = cupy.asarray(left_eye_mask)
            right_eye_mask = cupy.asarray(right_eye_mask)
            mounth_mask = cupy.asarray(mounth_mask)
        skin_image = image * mask * (1-left_eye_mask) * (1-right_eye_mask) * (1-mounth_mask)

        if self.device == 'GPU':
            rmin, rmax, cmin, cmax = bbox2_GPU(skin_image)
        else:
            rmin, rmax, cmin, cmax = bbox2_CPU(skin_image)

        cropped_skin_im = skin_image
        if rmin >= 0 and rmax >= 0 and cmin >= 0 and cmax >= 0 and rmax-rmin >= 0 and cmax-cmin >= 0:
            cropped_skin_im = skin_image[int(rmin):int(rmax), int(cmin):int(cmax)]

        if self.device == 'GPU':
            cropped_skin_im = cupy.asnumpy(cropped_skin_im)
            skin_image = cupy.asnumpy(skin_image)

        return cropped_skin_im, skin_image
