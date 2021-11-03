# TEST FUNCTIONS

import sys
import os

from torch.utils.data.dataloader import DataLoader


sys.path.insert(1, os.getcwd())


def test_metrics_calculation_for_lndb_dataset():
    from datasets import PATCH_DATASETS
    import numpy as np
    from datasets.utils import sitk2tensor
    from evaluations.evaluator import MetricEvaluator
    lndb_patch = PATCH_DATASETS["LNDbPatch32Dataset"](root_dir=None,
                                                      transform=sitk2tensor,
                                                      split='val')
    print("length of lndb_patch dataset", len(lndb_patch))
    lndb_dl = DataLoader(dataset=lndb_patch,
                         batch_size=36,
                         shuffle=False,
                         drop_last=False,
                         num_workers=4,
                         pin_memory=True)
    me = MetricEvaluator(metrics=["SSIM", "MSE", "PSNR"],
                         log_name='VAE3D32AUG',
                         version=18)
    metrics_dict = me.calc_metrics(dataloader=lndb_dl)
    for k, v in metrics_dict.items():
        print(f"{k}: mean value = {np.mean(v)}")
    pass


def test_recon_images_for_lndb_dataset():
    from datasets import PATCH_DATASETS
    import os.path as osp
    from datasets.utils import sitk2tensor
    from evaluations.evaluator import ReconEvaluator
    lndb_patch = PATCH_DATASETS["LNDbPatch32Dataset"](root_dir=None,
                                                      transform=sitk2tensor,
                                                      split='val')
    print("length of lndb_patch dataset", len(lndb_patch))
    lndb_dl = DataLoader(dataset=lndb_patch,
                         batch_size=36,
                         shuffle=False,
                         drop_last=False,
                         num_workers=4,
                         pin_memory=True)
    re = ReconEvaluator(vis_dir=osp.join(os.getcwd(), "evaluations/results/"),
                        log_name='VAE3D32AUG',
                        version=18)
    re(dataloader=lndb_dl)
    pass


if __name__ == "__main__":
    test_recon_images_for_lndb_dataset()
