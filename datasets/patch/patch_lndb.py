import os.path as osp

from configs.config_vars import DS_ROOT_DIR

from .patch_ds import PatchDataset

# TODO: not tested


class LNDbPatchDataset(PatchDataset):

    def __init__(self, *args, **kwargs):  # -> None
        if (not kwargs) or ('root_dir' not in kwargs) or (kwargs['root_dir'] is None):
            kwargs['root_dir'] = osp.join(DS_ROOT_DIR, 'LNDb/LNDb-patch/')
        super(LNDbPatchDataset, self).__init__(*args, **kwargs)

    def _get_patient_list(self, patch_name_list):
        patient_names = list(set([n.split('.')[0] for n in patch_name_list]))
        return patient_names

class LNDbPatch32Dataset(LNDbPatchDataset):

    def __init__(self, *args, **kwargs):  # -> None
        if (not kwargs) or ('root_dir' not in kwargs) or (kwargs['root_dir'] is None):
            kwargs['root_dir'] = osp.join(DS_ROOT_DIR, 'LNDb/LNDb-patch32/')
        super(LNDbPatch32Dataset, self).__init__(*args, **kwargs)
        
