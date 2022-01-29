import os.path as osp
from configs.config_vars import DS_ROOT_DIR
from .patch_ds import PatchDataset
from sklearn.model_selection import train_test_split


class StanfordRadiogenomicsPatchDataset(PatchDataset):

    def __init__(self, *args, **kwargs):  # -> None
        if (not kwargs) or ('root_dir' not in kwargs) or (kwargs['root_dir'] is None):
            kwargs['root_dir'] = osp.join(
                DS_ROOT_DIR, 'StanfordRadiogenomics/patch-32/')
        super(StanfordRadiogenomicsPatchDataset,
              self).__init__(ratio=0.3, *args, **kwargs)
        pass

    def _get_patient_list(self, patch_name_list):
        patient_names = list(set([n.split('.')[0] for n in patch_name_list]))
        return patient_names

    def set_split(self,
                  ratio=0.3,
                  split='all'):
        """ is overwritten, only splits train + val
        split should be one of the elem in split_set.
        this function takes care of the images that the dataset can load.
        - Takes care of potential preset train/val/test split.
        - And takes care of potential data leakage
        - Should output stable result each time.
        - Initialize self.patches
        """
        assert split in self.SPLIT_SET, "split invalid"
        if split == 'val':
            self.logger.info(
                "this dataset only has train/test splits, setting val as test")
            split = "test"
        patches_list = self._get_img_files()
        patient_list = self._get_patient_list(patches_list)
        if split == 'all':
            self.patches = patches_list
        else:
            idx = {'train': [], 'val': [], 'test': []}
            idx['train'], \
                idx['test'] = train_test_split(list(range(len(patient_list))),  # patient wise split
                                               test_size=ratio,  # default to be 0.1
                                               random_state=9001)
            patients = self._list_index(patient_list, idx[split])
            self.patches = self._get_patch_names(patients, patches_list)
            self.logger.info(
                f"patient split: train:{len(idx['train'])}, test:{len(idx['test'])}")
        self._split = split
        return split


class StanfordRadiogenomicsPatchAugDataset(StanfordRadiogenomicsPatchDataset):

    def __init__(self, *args, **kwargs):  # -> None
        if (not kwargs) or ('root_dir' not in kwargs) or (kwargs['root_dir'] is None):
            kwargs['root_dir'] = osp.join(
                DS_ROOT_DIR, 'StanfordRadiogenomics/patch-32-aug/')
        super(StanfordRadiogenomicsPatchAugDataset,
              self).__init__(ratio=0.3, *args, **kwargs)
        pass
