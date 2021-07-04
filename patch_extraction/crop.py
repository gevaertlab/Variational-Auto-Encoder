''' crop image to patches '''
import functools
from typing import Tuple
import SimpleITK as sitk
import numpy as np


@functools.singledispatch
def crop_patch(img: any,
               center_point: Tuple,
               size=(32, 32, 32)):
    raise NotImplementedError


@crop_patch.register(sitk.Image)
def _(img: sitk.Image,
      center_point: Tuple,
      size=(32, 32, 32)):
    """
    Crop image, simpleitk version
    Args:
        img ([sitk.Image]): Image to crop, note that should be uniformly spaced
        center_point ([tuple]): center position of the patch
        size (tuple, optional): [tuple]. Size of the patch in pixels. Defaults to (32, 32, 32).

    Returns:
        [numpy array (3D)]: patch cropped
    """
    start_pos = tuple(int(pos - diameter / 2) for pos,
                      diameter in zip(center_point, size))  # NOTE: modified
    # determine if there is anything out of bounds
    if min(start_pos) < 0 or \
            any([pos > size - patch_size for pos, size, patch_size in zip(start_pos, img.GetSize(), size)]):
        pad = sitk.ConstantPadImageFilter()
        # if lower bound
        if min(start_pos) < 0:
            # set lower bound to pad over the dimension that's smaller than 0
            pad.SetPadLowerBound((max(0 - pos, 0) for pos in start_pos))
            # now that the image will be padded, the start pos should move to at least 0
            start_pos = (max(0, pos) for pos in start_pos)
        if any([pos > size - patch_size for pos, size, patch_size in zip(start_pos, img.GetSize(), size)]):
            # if upper bound
            # note that though image size is not changed, but start_pos may be changed,
            # the direction(s) in which patch exceeded upper bound is/bare not changed!
            # we can still use same operations no problem

            # set upper bound to be the gap between end pad position and image size
            pad.SetPadUpperBound(tuple(max(0, pos + s - boundary)
                                       for pos, s, boundary in zip(start_pos, size, img.GetSize())))
            # no need to change start position
        padded_img = pad.Execute(img)
        return padded_img[start_pos[0]:start_pos[0]+size[0],
                          start_pos[1]:start_pos[1]+size[1],
                          start_pos[2]:start_pos[2]+size[2]]
    return img[start_pos[0]:start_pos[0]+size[0],
               start_pos[1]:start_pos[1]+size[1],
               start_pos[2]:start_pos[2]+size[2]]


@crop_patch.register(np.ndarray)
def _(img: np.ndarray,
      center_point: Tuple,
      size=(32, 32, 32)):
    """
    Crop image, numpy image version
    Returns:
        [numpy array (3D)]: patch cropped

    Args:
        img (np.ndarray): Image to crop, note that should be uniformly spaced
        center_point (Tuple): center position of the patch
        size (tuple, optional): Size of the patch in pixels. Defaults to (32, 32, 32).
    Returns:
        [numpy array (3D)]: patch cropped
    """
    start_pos = tuple(int(pos - diameter / 2) for pos,
                      diameter in zip(center_point, size))  # NOTE: modified
    end_pos = tuple(int(sp + s) for sp, s in zip(start_pos, size))

    # determine if there is anything out of bounds
    pad_width = ((0, 0), (0, 0), (0, 0))
    lower_width = list(max(-sp, 0) for sp in start_pos)
    upper_width = list(max(ep-s, 0) for ep, s in zip(end_pos, img.shape))
    pad_width = tuple((l, u) for l, u in zip(lower_width, upper_width))
    # if out of bounds, fix
    if any(any(w) for w in pad_width):
        # set new start pos
        start_pos = tuple(max(0, sp) for sp in start_pos)
        img = np.pad(img,
                     pad_width=pad_width,
                     mode='constant',
                     constant_values=0)
    return img[start_pos[0]:start_pos[0]+size[0],
               start_pos[1]:start_pos[1]+size[1],
               start_pos[2]:start_pos[2]+size[2]]
