""" util funcs and class related to dataset modifications """

import SimpleITK as sitk
import numpy as np
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split


def train_val_test_split(length: int,
                         ratio: float = 0.1,
                         random_state=9001):
    """
    Split dataset to train, val and test sets
    Args:
        length (int): length of the dataset to split (num_samples)
        ratio (float, optional): what's ratio of val and test set. Defaults to 0.1.
        random_state (int, optional): sklearn random_state. Defaults to 9001.

    Returns:
        array like: index for train, val and test set
    """
    indices = list(range(length))
    train_val_idx, test_idx = train_test_split(indices,
                                               test_size=ratio,
                                               random_state=random_state)
    train_idx, val_idx = train_test_split(train_val_idx,
                                          test_size=ratio/(1-ratio),
                                          random_state=random_state)
    return train_idx, val_idx, test_idx


class sitk2numpy(object):
    ''' Convert sitk image to numpy and add a channel as the first dimension (float numpy) '''

    def __init__(self):
        pass

    def __call__(self, sample):
        img, img_name = sample['image'], sample['image_name']
        npimg = sitk.GetArrayFromImage(img)
        npimg = np.expand_dims(npimg, axis=0).astype('float32')
        return {'image': npimg, 'image_name': img_name}


class np2tensor(object):
    ''' Convert numpy image to tensor (float tensor) '''

    def __init__(self):
        pass

    def __call__(self, sample):
        ''' 3D image, thus no need to swap axis '''
        img, img_name = sample['image'], sample['image_name']
        return {'image': torch.from_numpy(img), 'image_name': img_name}


sitk2tensor = transforms.Compose([sitk2numpy(), np2tensor()])
