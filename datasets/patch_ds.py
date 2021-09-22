""" patch dataset base """
import numpy as np
from torch.utils.data import ConcatDataset, Dataset
import SimpleITK as sitk
import os
from .utils import train_val_test_split


class PatchDataset(Dataset):
    """
    Patch Dataset with split
    Parent:
        Dataset (pytorch.data.Dataset)
    """

    def __init__(self,
                 root_dir: str,
                 transform=None,
                 split='train'):  # -> None
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
        # set to be relative path to patches from self.root_dir
        self.patches = []
        # the split of the dataset object can only be set through
        # set_split function, initialize self.images
        self.split = self.set_split(split)
        pass

    SPLIT_SET = {'all', 'train', 'val', 'test'}

    def set_split(self,
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
        if split == 'all':
            self.patches = patches_list
        else:
            idx = {'train': [], 'val': [], 'test': []}
            idx['train'], \
                idx['val'], \
                idx['test'] = train_val_test_split(len(patches_list),
                                                   ratio=0.1,
                                                   random_state=9001)
            self.patches = self._list_index(patches_list, idx[split])
        self._split = split
        return split

    def _get_img_files(self):
        file_lst = os.listdir(self.root_dir)
        return [file for file in file_lst if file.endswith('.nrrd')]

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx: int):
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