import os.path as osp

from configs.config_vars import DS_ROOT_DIR

from .patch_ds import PatchDataset

# TODO: not tested


class LNDbPatchDataset(PatchDataset):

    def __init__(self, *args, **kwargs):  # -> None
        if not kwargs or kwargs['root_dir'] is None:
            kwargs['root_dir'] = osp.join(DS_ROOT_DIR, 'LNDb/LNDb-patch/')
        super(LNDbPatchDataset, self).__init__(*args, **kwargs)
