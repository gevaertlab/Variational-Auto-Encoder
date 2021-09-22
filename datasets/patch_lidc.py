import os.path as osp

from configs.config_vars import DS_ROOT_DIR

from datasets.utils import train_val_test_split

from .label_dict import LABEL_DICT
from .patch_ds import PatchDataset


class LIDCPatchDataset(PatchDataset):

    def __init__(self, *args, **kwargs):
        """
        - Initialize the root_dir
        - Split functions
        """
        if not kwargs or kwargs['root_dir'] is None:
            kwargs['root_dir'] = osp.join(DS_ROOT_DIR, 'TCIA_LIDC/LIDC-patch/')
        super(LIDCPatchDataset, self).__init__(*args, **kwargs)
        pass

    def set_split(self, split):
        assert split in self.SPLIT_SET, "split invalid"
        patch_list = self._get_img_files()
        nodule_list = self._get_nodule_names(patch_list)
        idx = {}
        # NOTE: the train val test split will not change
        # AS LONG AS the total length of the nodule_list is not changed
        idx['train'], \
            idx['val'], \
            idx['test'] = train_val_test_split(len(nodule_list),
                                               ratio=0.1,
                                               random_state=9001)
        nodule_list_split = self._list_index(nodule_list, idx[split])
        self.patches = self._get_patch_names(nodule_name_list=nodule_list_split,
                                             patch_name_list=patch_list)
        pass


class LIDCPatchAugDataset(LIDCPatchDataset):

    def __init__(self, *args, **kwargs):  # -> None
        if not kwargs or kwargs['root_dir'] is None:
            kwargs['root_dir'] = osp.join(
                DS_ROOT_DIR, 'TCIA_LIDC/LIDC-patch-32_aug/')
        super(LIDCPatchDataset, self).__init__(*args, **kwargs)
        pass

    def _get_nodule_names(self, patch_name_list):
        """
        get nodule names from patch names with augmentations
        """
        result_list = [i.split('.') for i in patch_name_list]
        result_list = ['.'.join([i[0], i[1]]) for i in result_list]
        return list(set(result_list))

    def _get_patch_names(self, nodule_name_list, patch_name_list):
        """
        match patch names with nodule name list
        """
        result_list = []
        for patch_name in patch_name_list:
            if any([patch_name.startswith(nn) for nn in nodule_name_list]):
                result_list.append(patch_name)
        return result_list


class LIDCPatchLabelDataset(LIDCPatchDataset):

    def __init__(self, label_name: str, *args, **kwargs):  # -> None
        super(LIDCPatchDataset, self).__init__(*args, **kwargs)
        self.label_instance = LABEL_DICT[label_name]()
        pass

    def __getitem__(self, idx: int):
        image, image_name = super(
            LIDCPatchLabelDataset, self).__getitem__(self, idx=idx)
        label = self.label_instance.get_labels(image_name)
        return image, label
