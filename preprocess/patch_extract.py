''' 
This file extracts datasets to 3D patches and 
store the extracted patches in specified folder.
TODO: modify style
'''
from utils.visualization import visTempSitk
from utils.io import save_as_nrrd
from datasets.utils_lidc import LIDC
import pylidc as pl
import pandas as pd
import argparse
import os
import sys

import numpy as np
import SimpleITK as sitk

sys.path.append(os.getcwd())


class PatchExtract():
    """ 
    TODO: 
    1.comply with style, 
    2. customizable with CT datasets,  
    3. augmentation modules
    """

    def __init__(self, size, spacing=(1, 1, 1), augment_dict=None):  # -> None
        self.size = size
        self.spacing = spacing
        self.augment_dict = augment_dict
        pass

    def winsorizeNscale(self, img, winsorize_limits=(-1024, 3071), scale_limits=(0, 1)):
        image_array = sitk.GetArrayFromImage(img)
        # winsorize the array
        image_array = self.winsorize(image_array, limits=winsorize_limits)
        # scale the array
        image_array = self.scale(image_array, limits=scale_limits)
        # return to image
        new_img = sitk.GetImageFromArray(image_array)
        # restore other info
        new_img.CopyInformation(img)
        return new_img

    def scale(self, np_array, limits=(0, 1)):
        # scale the np array to be within range of limits
        org_range = np.max(np_array) - np.min(np_array)
        tgt_range = limits[1] - limits[0]
        distance = np.min(np_array) - limits[0]
        np_array = (np_array - distance) * (tgt_range / org_range)
        return np_array

    def winsorize(self, np_array: np.array, limits=(-1024, 3071)):
        np_array[np_array < limits[0]] = limits[0]
        np_array[np_array > limits[1]] = limits[1]
        return np_array

    def convertSpacing(self, img, center_point=None):
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetOutputDirection(img.GetDirection())
        resample.SetOutputOrigin(img.GetOrigin())
        new_spacing = self.spacing
        resample.SetOutputSpacing(new_spacing)

        orig_size = np.array(img.GetSize(), dtype=np.int)
        orig_spacing = img.GetSpacing()
        new_size = tuple(a * b / c for a, b,
                         c in zip(orig_size, orig_spacing, new_spacing))
        # Image dimensions are in integers
        new_size = np.ceil(new_size).astype(np.int)
        new_size = [int(s) for s in new_size]
        resample.SetSize(new_size)
        new_img = resample.Execute(img)
        # calculate center_point if there is input
        if not center_point is None:
            new_center_point = tuple(
                a * b / c for a, b, c in zip(center_point, orig_spacing, new_spacing))
            new_center_point = np.ceil(new_center_point).astype(
                np.int)  # Image dimensions are in integers
            return new_img, new_center_point
        return new_img

    def cropPatch(self, img, start_pos):
        # determine if there is any out of bounds
        if min(start_pos) < 0 or any([pos > size - patch_size for pos, size, patch_size in zip(start_pos, img.GetSize(), self.size)]):
            pad = sitk.ConstantPadImageFilter()
            # if lower bound
            if min(start_pos) < 0:
                # set lower bound to pad over the dimension that's smaller than 0
                pad.SetPadLowerBound((max(0 - pos, 0) for pos in start_pos))
                # now that the image will be padded, the start pos should move to at least 0
                start_pos = (max(0, pos) for pos in start_pos)
            if any([pos > size - patch_size for pos, size, patch_size in zip(start_pos, img.GetSize(), self.size)]):
                # if upper bound
                # note that though image size is not changed, but start_pos may be changed,
                # the direction(s) in which patch exceeded upper bound is/bare not changed!
                # we can still use same operations no problem

                # set upper bound to be the gap between end pad position and image size
                pad.SetPadUpperBound(tuple(max(0, pos + s - boundary)
                                           for pos, s, boundary in zip(start_pos, self.size, img.GetSize())))
                # no need to change start position
            padded_img = pad.Execute(img)
            return padded_img[start_pos[0]:start_pos[0]+self.size[0],
                              start_pos[1]:start_pos[1]+self.size[1],
                              start_pos[2]:start_pos[2]+self.size[2]]
        return img[start_pos[0]:start_pos[0]+self.size[0],
                   start_pos[1]:start_pos[1]+self.size[1],
                   start_pos[2]:start_pos[2]+self.size[2]]

    def execute(self, img, center_point):
        '''
        @param: img: sitk 3D CT image
        @param: center_point: (x,y,z) tuple of PIXEL POINT of center of the nodule i.e. img[x, y, z] = nodule
           Current preprocessing steps:
        1. Read in SITK image
        2. Convert spacing to (1, 1, 1)
        3. Winsorize to [-1024, 3071] and normalize to [0, 1]
        4. Crop to 3D patch
        '''
        # from utils.visualization import visTempSitk
        uniformly_spaced_img, new_center_point = self.convertSpacing(
            img, center_point)  # new_center_point (x, y, z)
        uniformly_spaced_scaled_img = self.winsorizeNscale(
            uniformly_spaced_img)
        start_pos = tuple(int(pos - diameter / 2) for pos,
                          diameter in zip(new_center_point, self.size))  # NOTE: modified
        patch = self.cropPatch(uniformly_spaced_scaled_img, start_pos)
        return patch
