''' 
extract patches:
    preprocessing
'''
import os
import sys
from typing import Tuple, Union

import numpy as np
import SimpleITK as sitk

sys.path.append(os.getcwd())


def winsorize_scale(img, winsorize_limits=(-1024, 3071), scale_limits=(0, 1)):
    image_array = sitk.GetArrayFromImage(img)
    # winsorize the array
    image_array = winsorize(image_array, limits=winsorize_limits)
    # scale the array
    image_array = scale(image_array, limits=scale_limits)
    # return to image
    new_img = sitk.GetImageFromArray(image_array)
    # restore other info
    new_img.CopyInformation(img)
    return new_img


def scale(np_array, limits=(0, 1)):
    # scale the np array to be within range of limits
    org_range = np.max(np_array) - np.min(np_array)
    tgt_range = limits[1] - limits[0]
    distance = np.min(np_array) - limits[0]
    np_array = (np_array - distance) * (tgt_range / org_range)
    return np_array


def winsorize(np_array: np.array, limits=(-1024, 3071)):
    np_array[np_array < limits[0]] = limits[0]
    np_array[np_array > limits[1]] = limits[1]
    return np_array


def _convert_point(center_point: Tuple,
                   orig_spacing: Tuple,
                   new_spacing: Tuple):
    """
    convert center point to new position (new spacing)
    Args:
        center_point ([tuple]): [old center coord]
        orig_spacing ([tuple]): [original spacing]
        new_spacing ([tuple]): [new spacing]

    Returns:
        [type]: [description]
    """
    new_center_point = tuple(a * b / c for a, b, c in
                             zip(center_point, orig_spacing, new_spacing))
    new_center_point = np.ceil(new_center_point).astype(np.int)
    # Image dimensions are in integers
    return new_center_point


def convert_spacing(img,
                    spacing=(1, 1, 1),
                    center_point=None):
    # TODO: new convert spacing function that's based on numpy array.
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    new_spacing = spacing
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
        if isinstance(center_point[0], (float, int)):
            new_center_point = _convert_point(center_point,
                                              orig_spacing, new_spacing)
        elif isinstance(center_point[0], (list, Tuple)):
            new_center_point = [_convert_point(
                c, orig_spacing, new_spacing) for c in center_point]
        return new_img, new_center_point
    return new_img


def transpose(img: np.ndarray,
              transpose_axis: Union[None, Tuple] = None):
    if not transpose_axis:
        return img
    else:
        return np.transpose(img, axes=transpose_axis)


def preprocess(img,
               center_point,
               spacing=(1, 1, 1),
               transpose_axis=None):
    '''
    @param: img: sitk 3D CT image
    @param: center_point: (x,y,z) tuple of PIXEL POINT of center of the nodule i.e. img[x, y, z] = nodule
        Current preprocessing steps:
    1. Read in SITK image
    2. Convert spacing to (1, 1, 1)
    3. Winsorize to [-1024, 3071] and normalize to [0, 1]
    return: np.array of uniformly_spaced_scaled_img
    NOTE: will return channel-last format image, 
    so may need potential conversion -> of image
    '''
    uniformly_spaced_img, new_center_point = convert_spacing(img,
                                                             spacing,
                                                             center_point)
    new_center_point = tuple(new_center_point)  # NOTE: convert to tuple
    uniformly_spaced_scaled_img = winsorize_scale(uniformly_spaced_img)
    npimg = sitk.GetArrayFromImage(uniformly_spaced_scaled_img) # (z, x, y)
    npimg = transpose(npimg, (1, 2, 0)) # (x, y, z)
    npimg = transpose(npimg, transpose_axis)
    return npimg, new_center_point
