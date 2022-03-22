import os
import os.path as osp
import sys

sys.path.insert(1, os.getcwd())

if __name__ == "__main__":
    # from datasets.label.label_stanfordradiogenomics import LabelStanfordRadiogenomicsVolume
    # lstfrv = LabelStanfordRadiogenomicsVolume()
    from datasets.ct.ct_stanfordradiogenomics import StanfordRadiogenomicsDataset
    sds = StanfordRadiogenomicsDataset(reset_info=True)
    sds[0]
    pass

