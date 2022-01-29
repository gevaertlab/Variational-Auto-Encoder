

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
                    LNDbPatchDataset,
                    LNDbPatch32Dataset,
                    LNDbPatch32AugDataset,
                    StanfordRadiogenomicsPatchDataset,
                    StanfordRadiogenomicsPatchAugDataset)

PATCH_DATASETS = {'LIDCPatchDataset': LIDCPatchDataset,
                  'LIDCPatchAugDataset': LIDCPatchAugDataset,
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
