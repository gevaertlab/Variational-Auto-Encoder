from .patch_ds import PatchDataset
from configs.config_vars import DS_ROOT_DIR
import os.path as osp


class LIDCPatchDataset(PatchDataset):

    def __init__(self, *args, **kwargs):  # -> None
        if not kwargs or kwargs['root_dir'] is None:
            kwargs['root_dir'] = osp.join(DS_ROOT_DIR, 'TCIA_LIDC/LIDC-patch/')
        super(LIDCPatchDataset, self).__init__(*args, **kwargs)
        pass


class LIDCPatch32Dataset(PatchDataset):

    def __init__(self, *args, **kwargs):  # -> None
        if not kwargs or kwargs['root_dir'] is None:
            kwargs['root_dir'] = osp.join(
                DS_ROOT_DIR, 'TCIA_LIDC/LIDC-patch/-32/')
        super(LIDCPatch32Dataset, self).__init__(*args, **kwargs)
        pass
