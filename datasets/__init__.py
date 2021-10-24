

# from .patch_ds import PatchDataset
from .ct import (CTDataset,
                 CTCachedDataset,
                 LIDCDataset,
                 LNDbDataset,
                 LIDCCachedDataset)

from .patch import (LIDCPatchAugDataset,
                    LIDCPatchDataset,
                    LIDCPatchLabelDataset,
                    LNDbPatchDataset)

PATCH_DATASETS = {'LIDCPatchDataset': LIDCPatchDataset,
                  'LIDCPatchAugDataset': LIDCPatchAugDataset,
                  'LNDbPatchDataset': LNDbPatchDataset,
                  'LIDCPatchLabelDataset': LIDCPatchLabelDataset}


CT_DATASETS = {'LIDCDataset': LIDCDataset,
               'LNDbDataset': LNDbDataset}
