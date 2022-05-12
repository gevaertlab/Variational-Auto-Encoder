import os.path as osp

from configs.config_vars import DS_ROOT_DIR
from datasets.label.label_dict import LABEL_DICT

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

    def _get_patient_list(self, patch_name_list):
        patient_names = list(set([n.split('.')[0] for n in patch_name_list]))
        return patient_names


class LIDCPatchAugDataset(LIDCPatchDataset):

    def __init__(self, *args, **kwargs):  # -> None
        if not kwargs or kwargs['root_dir'] is None:
            kwargs['root_dir'] = osp.join(
                DS_ROOT_DIR, 'TCIA_LIDC/LIDC-patch-32_aug/')
        super(LIDCPatchDataset, self).__init__(*args, **kwargs)
        pass

class LIDCPatchAugDebugDataset(LIDCPatchDataset):

    def __init__(self, length=60, *args, **kwargs):  # -> None
        if not kwargs or kwargs['root_dir'] is None:
            kwargs['root_dir'] = osp.join(
                DS_ROOT_DIR, 'TCIA_LIDC/LIDC-patch-32_aug/')
        super(LIDCPatchDataset, self).__init__(*args, **kwargs)
        self.length = length
        pass

    def __len__(self):
        return self.length

class LIDCPatchLabelDataset(LIDCPatchDataset):

    def __init__(self, label_name: str, *args, **kwargs):  # -> None
        super(LIDCPatchDataset, self).__init__(*args, **kwargs)
        self.label_instance = LABEL_DICT[label_name]()
        pass

    def __getitem__(self, idx: int):
        image, image_name = super(
            LIDCPatchLabelDataset, self).__getitem__(idx=idx)
        label = self.label_instance.get_labels(image_name)
        return image, label
