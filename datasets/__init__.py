

# from .patch_ds import PatchDataset

from .ct import (CTDataset,
                 CTCachedDataset,
                 LIDCDataset,
                 LNDbDataset,
                 LIDCCachedDataset,
                 StanfordRadiogenomicsDataset)

from .patch import (LIDCPatchAugDataset,
                    LIDCPatchDataset,
                    LIDCPatchLabelDataset,
                    LIDCPatchAugDebugDataset,
                    LNDbPatchDataset,
                    LNDbPatch32Dataset,
                    LNDbPatch32AugDataset,
                    StanfordRadiogenomicsPatchDataset,
                    StanfordRadiogenomicsPatchAugDataset)

PATCH_DATASETS = {'LIDCPatchDataset': LIDCPatchDataset,
                  'LIDCPatchAugDataset': LIDCPatchAugDataset,
                  'LIDCPatchLabelDataset': LIDCPatchLabelDataset,
                  'LIDCPatchAugDebugDataset': LIDCPatchAugDebugDataset,
                  'LNDbPatchDataset': LNDbPatchDataset,
                  'LIDCPatchLabelDataset': LIDCPatchLabelDataset,
                  'LNDbPatch32Dataset': LNDbPatch32Dataset,
                  'LNDbPatch32AugDataset': LNDbPatch32AugDataset,
                  'StanfordRadiogenomicsPatchDataset': StanfordRadiogenomicsPatchDataset,
                  'StanfordRadiogenomicsPatchAugDataset': StanfordRadiogenomicsPatchAugDataset}

CT_DATASETS = {'LIDCDataset': LIDCDataset,
               'LNDbDataset': LNDbDataset,
               'LIDCCachedDataset': LIDCCachedDataset,
               'StanfordRadiogenomicsDataset': StanfordRadiogenomicsDataset}
