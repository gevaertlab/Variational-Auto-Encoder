from .patch_ds import PatchDataset
from .patch_lidc import LIDCPatchAugDataset, LIDCPatchDataset
from .patch_lndb import LNDbPatchDataset


REGISTERED_DATASETS = {'PatchDataset': PatchDataset,
                       'LIDCPatchDataset': LIDCPatchDataset,
                       'LIDCPatchAugDataset': LIDCPatchAugDataset,
                       'LNDbPatchDataset': LNDbPatchDataset}
