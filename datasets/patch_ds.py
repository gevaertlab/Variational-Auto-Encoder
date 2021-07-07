""" patch dataset base """
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

    def __init__(self, root_dir: str, transform=None, split='train'):  # -> None
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
        self.idx_dict['train'], self.idx_dict['val'], self.idx_dict['test'] = train_val_test_split(
            len(self.img_lst))
        if self.split == 'all':
            pass
        elif split in ['train', 'val', 'test']:
            self.img_lst = [self.img_lst[i] for i in range(
                len(self.img_lst)) if i in self.idx_dict[self.split]]
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
