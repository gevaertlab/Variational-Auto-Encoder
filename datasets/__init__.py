

# from .patch_ds import PatchDataset
from .ct_lidc import LIDCDataSet

from .patch_lidc import (LIDCPatchAugDataset,
                         LIDCPatchDataset,
                         LIDCPatchLabelDataset)
from .patch_lndb import LNDbPatchDataset

PATCH_DATASETS = {'LIDCPatchDataset': LIDCPatchDataset,
                  'LIDCPatchAugDataset': LIDCPatchAugDataset,
                  'LNDbPatchDataset': LNDbPatchDataset,
                  'LIDCPatchLabelDataset': LIDCPatchLabelDataset}


CT_DATASETS = {'LIDCDataset': LIDCDataSet}
