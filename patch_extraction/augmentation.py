''' some functions for augmentation purposes '''
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy import ndimage
from .util import rotate_3d_coord, get_center_np
import random


class Augmentation:

    def __init__(self, params, num_axis=3):
        """
        Args:
            params: [(process, times)], process = [(name_of_func, params of func)]
            num_axis([int]): assume 3d image
        """
        self.params = params
        self.axis = list(range(num_axis))
        self.axes_lst = list(itertools.combinations(self.axis, 2))
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
    def _shift_cp(point, value, axis):
        """calculate shift point

        Args:
            point ([tuple]): [coord of point to shift]
            value ([int]): [number of pixels to move]
            axis ([int (for now)]): [the direction defined by axis]

        Returns:
            [tuple]: [new coord]
        """
        point[axis] += value
        return point

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
        for process, times in self.params:
            # initialize random seed for each process
            curr_seed = random_seed
            for time in range(times):
                aug_img, aug_cp = self.augment(img,
                                               center_point,
                                               process,
                                               random_seed=curr_seed)
                yield aug_img, aug_cp
                if curr_seed:
                    curr_seed += 1
                else:
                    curr_seed = None

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
