''' some functions for augmentation purposes '''
import itertools
import os
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from .util import get_center_np, rotate_3d_coord


class Augmentation:

    def __init__(self, params, num_axis=3, debug=False):
        """
        Args:
            params: [(process, times)], process = [(name_of_func, params of func)]
            num_axis([int]): assume 3d image
        """
        self.params = params
        self.axis = list(range(num_axis))
        self.axes_lst = list(itertools.combinations(self.axis, 2))
        self.debug = debug  # debug flag
        pass

    def shift(self, img, center_point, params, random_seed=None):
        """
        shift the center_point so that the patch extracted would 
        have the nodule not exactly fixed in center
        Args:
            img ([np.array]): 3D
            center_point ([tuple]): center point of patch to be extracted

        Returns:
            [tuple]: [(img, center_point))]
        """
        if random_seed:
            random.seed(random_seed)
        value = random.randint(params['range'][0],
                               params['range'][1])
        axis = random.randint(0, 2)
        return (img, self._shift_cp(center_point, value, axis))

    @staticmethod
    def _shift_cp(point: Tuple, value: int, axis: int):
        """calculate shift point

        Args:
            point ([tuple]): [coord of point to shift]
            value ([int]): [number of pixels to move]
            axis ([int (for now)]): [the direction defined by axis]

        Returns:
            [tuple]: [new coord]
        """
        new_point = list(point)
        new_point[axis] += value
        new_point = tuple(new_point)
        return new_point

    def rotate(self, img, center_point, params, random_seed=None):
        """
        rotate the image so that patch will be a bit different
        Args:
            img ([np.array]): 3D
            center_point ([tuple]): center point of patch to be extracted
        Returns:
            list of tuple: [(img, center_point), (img, center_point), ...]
        """
        origin = get_center_np(img)
        if random_seed:
            random.seed(random_seed)
        axis_plane = random.choice(self.axes_lst)
        angle = random.randint(params['range'][0],
                               params['range'][1])

        img_rotated = ndimage.rotate(input=img,
                                     angle=angle,
                                     axes=axis_plane,
                                     reshape=False)  # not reshape to find point
        rotated_center_point = rotate_3d_coord(origin,
                                               center_point,
                                               angle,
                                               axis_plane)

        return (img_rotated, rotated_center_point)

    def augment_generator(self, img, center_point, random_seed=None):
        """
        augmentation process generator, yield a case each time, according to params
        Args:
            img ([np.array]): 3D
            center_point ([tuple]): center point of patch to be extracted
            random_seed ([int or None], optional): [random seed for each function]. Defaults to None.
        """
        # return original image first
        yield img, center_point
        # for a augmentation process, repeat how many times.
        # for process, repeat in self.params:
        process, repeat = self.params
        # initialize random seed for each process
        curr_seed = random_seed
        for time in range(repeat):
            aug_img, aug_cp = self.augment(img,
                                           center_point,
                                           process,
                                           random_seed=curr_seed)
            # potential debugging
            if self.debug:
                # debug mode
                self.vis_patch_aug([img, center_point],
                                   [aug_img, aug_cp],
                                   time)
            yield aug_img, aug_cp
            if curr_seed:
                curr_seed += 1
            else:
                curr_seed = None
        pass

    def augment(self, img, center_point, process, random_seed=None):
        # initialize img and center point each time
        aug_img, aug_cp = img, center_point
        if process:  # if process list is empty, do nothing
            for func_name, func_params in process:
                # each step of the process
                aug_img, aug_cp = getattr(self, func_name)(aug_img,
                                                           aug_cp,
                                                           func_params,
                                                           random_seed)
        return aug_img, aug_cp

    def vis_patch_aug(self,
                      origin_img_n_center,
                      aug_img_n_center,
                      time,
                      vis_dir="/labs/gevaertlab/users/yyhhli/code/debug/",):
        [org_img, org_cp], [aug_img, aug_cp] = origin_img_n_center, aug_img_n_center
        fig, ax = plt.subplots(2, 1)
        ax[0].imshow(org_img)
        ax[0].plot(org_cp)
        ax[0].annotate(org_cp, "center_point")
        ax[1].imshow(aug_img)
        ax[1].plot(aug_cp)
        ax[1].annotate(aug_cp, "center_point")
        plt.savefig(os.path.join(vis_dir, f'aug_{str(time)}'),
                    dpi=300)
        pass


def calc_crop_size(patch_size: Tuple, aug_params: List):
    """
    calculate crop size before augmentation
    NOTE: this size calculation highly depends on the type and 
    parameters of the augmentation process
    return size
    """
    new_range = 0

    # shift
    shift = [op for op in aug_params[0] if op[0] == 'shift']
    shift_ranges = sum([list(op[1]['range']) for op in shift], [])
    max_range = 0 if not shift_ranges else int(np.max(np.abs(shift_ranges)))
    new_range += max_range

    # rotation
    extended = np.ceil(np.max(patch_size) * 1.732).astype(int) - \
        np.min(patch_size)  # sqrt3
    new_range += extended

    # make new size
    new_size = tuple(map(sum, zip(patch_size, [new_range]*len(patch_size))))
    return new_size
