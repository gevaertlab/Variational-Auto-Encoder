# stanford CT patch dataset, moved from VAE

""" CT Dataset for NSCLC """
import os
import os.path as osp
import sys

sys.path.insert(1, "/labs/gevaertlab/user/yyhhli/vae/")

from applications.datasets.patch.patch_stanfordardiogenomics import StanfordRadiogenomicsPatchAugDataset
