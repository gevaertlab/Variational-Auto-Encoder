import sys
import argparse
import numpy as np
import os
import os.path as osp
from torch.utils.data.dataloader import DataLoader

sys.path.insert(1, "/labs/gevaertlab/users/yyhhli/code/vae/")


def recon_vis_eval():

    parser = argparse.ArgumentParser(
        description='Recon visualization in STF dataset')
    parser.add_argument("--version", "-v",
                        default=60)
    parser.add_argument("--log_name", "-ln", default="VAE3D32AUG")
    args = parser.parse_args()

    from datasets import PATCH_DATASETS
    from datasets.utils import sitk2tensor
    from evaluations.evaluator import ReconEvaluator

    lndb_patch = PATCH_DATASETS["LNDbPatch32Dataset"](root_dir=None,
                                                      transform=sitk2tensor,
                                                      split='val')
    print("length of stanford_radiogenomics dataset",
          len(lndb_patch))
    lndb_dl = DataLoader(dataset=lndb_patch,
                         batch_size=36,
                         shuffle=False,
                         drop_last=False,
                         num_workers=4,
                         pin_memory=True)
    re = ReconEvaluator(vis_dir=osp.join(os.getcwd(), "evaluations/results/"),
                        log_name=args.log_name,
                        version=args.version)

    re(dataloader=lndb_dl, dl_params={
       "name": "LNDb", 'shuffle': False, 'drop_last': False})
    pass


if __name__ == "__main__":
    recon_vis_eval()
