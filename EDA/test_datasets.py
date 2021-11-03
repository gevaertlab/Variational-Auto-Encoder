# TEST FUNCTIONS

import sys
import os


sys.path.insert(1, os.getcwd())


def test_lidc_dataset():
    from datasets import LIDCDataset
    lidc = LIDCDataset()
    info = lidc.get_info(24)
    print(info)
    pass


def test_lndb_dataset():
    from datasets import LNDbDataset
    lndb = LNDbDataset()
    info = lndb.get_info(24)
    print(info)
    pass


def test_patch_lndb():
    from patch_extraction.patch_extract import PatchExtract
    from datasets import LNDbDataset
    lndb = LNDbDataset()
    pe = PatchExtract(patch_size=(32, 32, 32), dataset=lndb,
                      augmentation_params=None, debug=True)
    pe.load_extract_ds(
        save_dir="/labs/gevaertlab/data/lung cancer/LNDb/LNDb-patch32",
        overwrite=True)
    pe.vis_ds(dataset_dir="/labs/gevaertlab/data/lung cancer/LNDb/LNDb-patch32",
              vis_dir="/labs/gevaertlab/data/lung cancer/LNDb/vis_LNDb-patch32")
    pass


def test_patch_extraction_lidc():
    from patch_extraction.patch_extract import PatchExtract
    from datasets import LIDCDataset
    lidc = LIDCDataset()
    pe = PatchExtract(patch_size=(64, 64, 64), dataset=lidc,
                      augmentation_params=None, debug=True)
    pe.load_extract_ds(
        save_dir="/labs/gevaertlab/data/lung cancer/TCIA_LIDC/LIDC-test-64",
        overwrite=True)
    # pe.vis_ds(dataset_dir="/labs/gevaertlab/data/lung cancer/TCIA_LIDC/LNDb-patch32",
    #           vis_dir="/labs/gevaertlab/data/lung cancer/LNDb/vis_LNDb-patch32")
    pass


def test_lndb_patch_dataset():
    from datasets import PATCH_DATASETS
    from datasets.utils import sitk2tensor
    from evaluations.evaluator import MetricEvaluator
    lndb_patch = PATCH_DATASETS["LNDbPatch32Dataset"](root_dir=None,
                                                      transform=sitk2tensor,
                                                      split='train')
    print(len(lndb_patch))
    data = lndb_patch[1]
    pass


def test_lidc_patch_dataset_split():
    from datasets import PATCH_DATASETS
    from datasets.utils import sitk2tensor
    from evaluations.evaluator import MetricEvaluator
    lidc_patch = PATCH_DATASETS["LIDCPatchDataset"](root_dir=None,
                                                    transform=sitk2tensor,
                                                    split='train')
    data = lidc_patch[1]
    pass


if __name__ == '__main__':
    test_lndb_patch_dataset()
