""" patch dataset base """
import os
from SimpleITK.SimpleITK import Not
from abc import ABCMeta, abstractmethod
import numpy as np
import SimpleITK as sitk
from datasets.utils import train_val_test_split
from datasets.ct import CTDataset
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms


class PatchDataset(Dataset):
    """
    Patch Dataset with split
    Parent:
        Dataset (pytorch.data.Dataset)
    """
    __metaclass__ = ABCMeta

    def __init__(self,
                 root_dir: str,
                 transform=None,
                 split='train',
                 ratio=0.1):  # -> None
        """
        Args:
            root_dir (str): root directory of the dataset
            transform ([callable, None], optional): the transformation function will be applied to 
            samples in __getitem__ function. Defaults to None.
            split (str, optional): 'train' or 'val' or 'test'. Defaults to 'train'.
            need to implement two functions for datasets:
            _get_nodule_names(patch_name_list) and _get_patch_names(nodule_name_list, patch_name_list)
        """
        super(PatchDataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        # set to be relative path to patches from self.root_dir
        self.patches = []
        # the split of the dataset object can only be set through
        # set_split function, initialize self.images
        self.split = self.set_split(ratio=ratio, split=split)
        pass

    SPLIT_SET = {'all', 'train', 'val', 'test'}

    def set_split(self,
                  ratio=0.1,
                  split='all'):
        """should be overwritten.
        split should be one of the elem in split_set.
        this function takes care of the images that the dataset can load.
        - And takes care of potential data leakage
        - Takes care of potential preset train/val/test split.
        - Should output stable result each time.
        - Initialize self.patches
        """
        assert split in self.SPLIT_SET, "split invalid"
        patches_list = self._get_img_files()
        patient_list = self._get_patient_list(patches_list)
        if split == 'all':
            self.patches = patches_list
        else:
            idx = {'train': [], 'val': [], 'test': []}
            idx['train'], \
                idx['val'], \
                idx['test'] = train_val_test_split(len(patient_list),  # patient wise split
                                                   ratio=ratio,  # default to be 0.1
                                                   random_state=9001)
            patients = self._list_index(patient_list, idx[split])
            self.patches = self._get_patch_names(patients, patches_list)
            # self.patches = self._list_index(patches_list, idx[split])
        self._split = split
        return split

    def _get_img_files(self):
        file_lst = os.listdir(self.root_dir)
        return [file for file in file_lst if file.endswith('.nrrd')]

    @abstractmethod
    def _get_patient_list(self, patch_name_list):
        """list all the patients for setting split from the patch in the list"""
        raise NotImplementedError("Must override")

    @staticmethod
    def _get_nodule_names(patch_name_list):
        """
        get nodule names from patch names with AUGMENTATION
        """
        result_list = [i.split('.') for i in patch_name_list]
        result_list = ['.'.join([i[0], i[1]]) for i in result_list]
        return list(set(result_list))

    @staticmethod
    def _get_patch_names(query_name_list, patch_name_list):
        """
        match patch names with nodule name list
        """
        result_list = []
        for patch_name in patch_name_list:
            if any([patch_name.startswith(nn) for nn in query_name_list]):
                result_list.append(patch_name)
        return result_list

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx: int):
        """ gets sitk image instance and image name """
        img_name = self.patches[idx]
        img = sitk.ReadImage(os.path.join(self.root_dir, img_name))
        sample = {'image': img, 'image_name': img_name}
        # apply transformation
        if self.transform is not None:
            sample = self.transform(sample)
        return sample['image'], sample['image_name']  # debug modified

    @staticmethod
    def _list_index(lst, idx_lst):
        sorted_idx = sorted(idx_lst)
        return list(np.array(lst)[sorted_idx])


class PatchDynamicDataset(Dataset):  # TODO: rethink the usage

    def __init__(self,
                 ct_dataset: CTDataset,
                 transform: transforms):
        self.ct_dataset = ct_dataset
        self.transform = transform
        pass

    def __len__(self):
        return self.ct_dataset.__len__()

    def __getitem__(self, idx: int):
        raise NotImplementedError
