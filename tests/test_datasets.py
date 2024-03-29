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
        save_dir="/labs/gevaertlab/data/lungcancer/LNDb/LNDb-patch32",
        overwrite=True)
    pe.vis_ds(dataset_dir="/labs/gevaertlab/data/lungcancerLNDb/LNDb-patch32",
              vis_dir="/labs/gevaertlab/data/lungcancer/LNDb/vis_LNDb-patch32")
    pass


def test_patch_extraction_lidc():
    from patch_extraction.patch_extract import PatchExtract
    from datasets import LIDCDataset
    lidc = LIDCDataset()
    pe = PatchExtract(patch_size=(64, 64, 64), dataset=lidc,
                      augmentation_params=None, debug=True)
    pe.load_extract_ds(
        save_dir="/labs/gevaertlab/data/lungcancerIA_LIDC/LIDC-test-64",
        overwrite=True)
    # pe.vis_ds(dataset_dir="/labs/gevaertlab/data/lungcancer/TCIA_LIDC/LNDb-patch32",
    #           vis_dir="/labs/gevaertlab/data/lungcancer/LNDb/vis_LNDb-patch32")
    pass


def test_lndb_patch_dataset():
    from datasets import PATCH_DATASETS
    from datasets.utils import sitk2tensor
    lndb_patch = PATCH_DATASETS["LNDbPatch32Dataset"](root_dir=None,
                                                      transform=sitk2tensor,
                                                      split='train')
    print(len(lndb_patch))
    data = lndb_patch[1]
    print(data)
    pass


def test_lidc_patch_dataset_split():
    from datasets import PATCH_DATASETS
    from datasets.utils import sitk2tensor
    lidc_patch = PATCH_DATASETS["LIDCPatchDataset"](root_dir=None,
                                                    transform=sitk2tensor,
                                                    split='train')
    data = lidc_patch[1]
    print(data)
    pass


def test_lidc_label():
    from datasets import PATCH_DATASETS
    from datasets.utils import sitk2tensor
    from datasets import LNDbDataset
    lndb = LNDbDataset()
    info = lndb.get_info(24)
    print(info)
    lndb_patch = PATCH_DATASETS["LNDbPatch32Dataset"](root_dir=None,
                                                      transform=sitk2tensor,
                                                      split='train')
    data = lndb_patch[1]
    print(data)
    pass


def test_lidc_split():
    from datasets import PATCH_DATASETS
    from datasets.utils import sitk2tensor
    lidc_patch = PATCH_DATASETS["LIDCPatchDataset"](root_dir=None,
                                                    transform=sitk2tensor,
                                                    split='train')
    data = lidc_patch[1]
    print(data)
    pass


def test_lidc_split():
    from datasets import PATCH_DATASETS
    from datasets.utils import sitk2tensor
    lndb_patch = PATCH_DATASETS["LNDbPatch32Dataset"](root_dir=None,
                                                      transform=sitk2tensor,
                                                      split='train')
    len(lndb_patch)
    pass


def test_stanfordradiogenomics_dataset():
    sys.path.insert(1, os.getcwd())
    from datasets.ct.ct_stanfordradiogenomics import StanfordRadiogenomicsDataset
    stanfordradiogenomics = StanfordRadiogenomicsDataset()
    data = stanfordradiogenomics[1]
    f = stanfordradiogenomics._get_files()
    stanfordradiogenomics._set_ds_info()
    pass


def test_stanford_label_dataset():
    sys.path.insert(1, os.getcwd())
    from baselines.cnn.dataset import StanfordLabelDataset
    from datasets.utils import sitk2tensor
    from datasets.label.label_stanfordradiogenomics import LabelStfAJCC
    slds = StanfordLabelDataset(LabelStfAJCC(), transform=sitk2tensor)
    data = slds[0]
    print(data)
    from torch.utils.data import DataLoader
    dl = DataLoader(slds, batch_size=1, shuffle=False)
    data = next(iter(dl))
    print(data)
    pass


if __name__ == '__main__':
    test_stanford_label_dataset()
