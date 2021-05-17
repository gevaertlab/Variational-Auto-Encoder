""" Dataset functions """
import os

import numpy as np
import SimpleITK as sitk
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms


def train_val_test_split(length: int, ratio: float=0.1, random_state=9001):
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


class PatchDataset(Dataset):
    """
    Patch Dataset with split
    Parent:
        Dataset (pytorch.data.Dataset)
    """
    
    def __init__(self, root_dir: str, transform = None, split='train'): #  -> None
        """
        Args:
            root_dir (str): root directory of the dataset
            transform ([callable, None], optional): the transformation function will be applied to 
            samples in __getitem__ function. Defaults to None.
            split (str, optional): 'train' or 'val' or 'test'. Defaults to 'train'.
        """
        super(PatchDataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.img_lst = self.__getImgName__()
        self.split = split
        self.idx_dict = {}
        self.idx_dict['train'], self.idx_dict['val'], self.idx_dict['test'] = train_val_test_split(len(self.img_lst))
        if self.split == 'all':
            pass
        elif split in ['train', 'val', 'test']:
            self.img_lst = [self.img_lst[i] for i in range(len(self.img_lst)) if i in self.idx_dict[self.split]]
        pass

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


class LIDCPatchDataset(PatchDataset):

    def __init__(self, *args, **kwargs): # -> None
        if not kwargs or kwargs['root_dir'] is None:
            kwargs['root_dir'] = '/labs/gevaertlab/data/lung cancer/TCIA_LIDC/LIDC-patch/'
        super(LIDCPatchDataset, self).__init__(*args, **kwargs)
        pass


class LIDCPatch32Dataset(PatchDataset):
    
    def __init__(self, *args, **kwargs): # -> None
        if not kwargs or kwargs['root_dir'] is None:
            kwargs['root_dir'] = '/labs/gevaertlab/data/lung cancer/TCIA_LIDC/LIDC-patch-32/'
        super(LIDCPatch32Dataset, self).__init__(*args, **kwargs)
        pass


class LNDbPatchDataset(PatchDataset):

    def __init__(self, *args, **kwargs): # -> None
        if not kwargs or kwargs['root_dir'] is None:
            kwargs['root_dir'] = '/labs/gevaertlab/data/lung cancer/LNDb/LNDb-patch/'
        super(LNDbPatchDataset, self).__init__(*args, **kwargs)


def getCombinedDS(*args, **kwargs):
    '''
    get combined dataset 
    ! not used in this version
    @param: root_dir: root directory of dataset, must be None
    @param: transformation: transformation of images
    @param: split: either 'train', 'val' or 'test'
    '''
    datasets = [LIDCPatchDataset(*args, **kwargs), LNDbPatchDataset(*args, **kwargs)]
    return ConcatDataset(datasets)


class LNDbDebugDataset(PatchDataset):

    def __init__(self, size: int=None, *args, **kwargs): #  -> None
        '''
        Added a option for selecting image size, just for debug purpose, 
        if not specified, load all.
        '''
        if not kwargs or kwargs['root_dir'] is None:
            kwargs['root_dir'] = '/labs/gevaertlab/data/lung cancer/LNDb/LNDb-patch/'
        super(PatchDataset, self).__init__(*args, **kwargs)
        self.root_dir = kwargs['root_dir']
        self.transform = kwargs['transform']
        self.img_lst = self.__getImgName__(size=size)


    def __getImgName__(self, size: int):
        file_lst = os.listdir(self.root_dir)
        img_lst = [file for file in file_lst if file.endswith('.nrrd')]
        if not size is None:
            return img_lst[:size]
        else:
            return img_lst


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
