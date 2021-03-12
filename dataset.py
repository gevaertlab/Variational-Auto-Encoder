import os
import re
from time import time

import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PatchDataset(Dataset):

    def __init__(self, root_dir, transform=None) -> None:
        super(PatchDataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.img_lst = self.__getImgName__()

    def __getImgName__(self):
        file_lst = os.listdir(self.root_dir)
        return [file for file in file_lst if file.endswith('.nrrd')]

    def __len__(self):
        return len(self.img_lst)

    def __getitem__(self, idx: int):
        img_name = self.img_lst[idx]
        img = sitk.ReadImage(os.path.join(self.root_dir, img_name))
        sample = {'image': img, 'image_name': img_name}
        # apply transformation
        if self.transform is not None:
            sample = self.transform(sample)
        return sample['image'], sample['image_name']  # debug modified


class LIDCPathDataset(PatchDataset):

    def __init__(self, root_dir=None, transform=None) -> None:
        if root_dir is None:
            root_dir = '/labs/gevaertlab/data/lung cancer/TCIA_LIDC/LIDC-patch/'
        super().__init__(root_dir, transform=transform)


class LNDbPathDataset(PatchDataset):

    def __init__(self, root_dir=None, transform=None) -> None:
        if root_dir is None:
            root_dir = '/labs/gevaertlab/data/lung cancer/LNDb/LNDb-patch/'
        super().__init__(root_dir, transform=transform)


class LNDbDebugDataset(PatchDataset):

    def __init__(self, size: int, root_dir=None, transform=None) -> None:
        if root_dir is None:
            root_dir = '/labs/gevaertlab/data/lung cancer/LNDb/LNDb-patch/'
        super(PatchDataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.img_lst = self.__getImgName__(size=size)


    def __getImgName__(self, size: int):
        file_lst = os.listdir(self.root_dir)
        img_lst = [file for file in file_lst if file.endswith('.nrrd')]
        return img_lst[:size]

class Sitk2Numpy(object):
    ''' Convert sitk image to numpy and add a channel as the first dimension (float numpy) '''

    def __init__(self):
        pass

    def __call__(self, sample):
        img, img_name = sample['image'], sample['image_name']
        npimg = sitk.GetArrayFromImage(img)
        npimg = np.expand_dims(npimg, axis=0).astype('float32')
        return {'image': npimg, 'image_name': img_name}


class Np2Tensor(object):
    ''' Convert numpy image to tensor (float tensor) '''

    def __init__(self):
        pass

    def __call__(self, sample):
        ''' 3D image, thus no need to swap axis '''
        img, img_name = sample['image'], sample['image_name']
        return {'image': torch.from_numpy(img), 'image_name': img_name}


sitk2tensor = transforms.Compose([Sitk2Numpy(), Np2Tensor()])
