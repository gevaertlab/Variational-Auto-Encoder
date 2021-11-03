

# from .patch_ds import PatchDataset

from .ct import (CTDataset,
                 CTCachedDataset,
                 LIDCDataset,
                 LNDbDataset,
                 LIDCCachedDataset)

from .patch import (LIDCPatchAugDataset,
                    LIDCPatchDataset,
                    LIDCPatchLabelDataset,
                    LNDbPatchDataset,
                    LNDbPatch32Dataset)

PATCH_DATASETS = {'LIDCPatchDataset': LIDCPatchDataset,
                  'LIDCPatchAugDataset': LIDCPatchAugDataset,
                  'LNDbPatchDataset': LNDbPatchDataset,
                  'LIDCPatchLabelDataset': LIDCPatchLabelDataset,
                  'LNDbPatch32Dataset': LNDbPatch32Dataset}

CT_DATASETS = {'LIDCDataset': LIDCDataset,
               'LNDbDataset': LNDbDataset,
               'LIDCCachedDataset': LIDCCachedDataset}
