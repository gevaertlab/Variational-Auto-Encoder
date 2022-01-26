

from typing import List

from torch.utils.data.dataset import Dataset
from . import PATCH_DATASETS
from utils.python_logger import get_logger


def get_concat_dataset(patch_datasets: List):
    """
    concatenate patch datasets
    Parent:
        Dataset (pytorch.data.Dataset)
    """
    patch_datasets = {pd:PATCH_DATASETS[pd] for pd in patch_datasets}

    class ConcatPatchDataset(Dataset):

        def __init__(self,
                     root_dir,
                     transform,
                     split):
            super(ConcatPatchDataset, self).__init__()
            self.logger = get_logger(cls_name=self.__class__.__name__)
            self.patch_datasets = {}
            arg_dict = {"root_dir": root_dir,
                        "transform": transform,
                        "split": split}
            for name, arg in arg_dict.items():
                if not isinstance(arg, list):
                    arg_dict[name] = [arg] * len(patch_datasets)
                else:
                    assert len(arg) == len(
                        patch_datasets), "argument lengths differ"
            for i, (name, ds) in enumerate(patch_datasets.items()):
                self.patch_datasets[name] = ds(root_dir=arg_dict["root_dir"][i],
                                               transform=arg_dict["transform"][i],
                                               split=arg_dict["split"][i])

        def __len__(self):
            return sum([len(ds) for ds in self.patch_datasets.values()])

        def __getitem__(self, index):
            index = index % self.__len__()
            for i, (name, ds) in enumerate(self.patch_datasets.items()):
                if len(ds) > index:
                    return ds[index]
                else:
                    index -= len(ds)
            return
    return ConcatPatchDataset
