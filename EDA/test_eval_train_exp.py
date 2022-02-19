
import sys
import os
import os.path as osp

from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(1, os.getcwd())


def test_metrics_calculation_for_lidc_dataset():
    from datasets import PATCH_DATASETS
    import numpy as np
    from datasets.utils import sitk2tensor
    from evaluations.evaluator import MetricEvaluator
    lidc_patch = PATCH_DATASETS["LIDCPatchAugDataset"](root_dir=None,
                                                       transform=sitk2tensor,
                                                       split='val')
    print("length of lidc_patch dataset", len(lidc_patch))
    lidc_dl = DataLoader(dataset=lidc_patch,
                         batch_size=36,
                         shuffle=False,
                         drop_last=False,
                         num_workers=4,
                         pin_memory=True)
    me = MetricEvaluator(metrics=["SSIM", "MSE", "PSNR"],
                         log_name='VAE3D32AUG',
                         version=39)
    metrics_dict = me.calc_metrics(dataloader=lidc_dl)
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
                        version=39)
    re(dataloader=lndb_dl)
    pass


def test_recon_images_for_stf_dataset(version):
    from datasets import PATCH_DATASETS
    import os.path as osp
    from datasets.utils import sitk2tensor
    from evaluations.evaluator import ReconEvaluator
    import numpy as np
    from evaluations.evaluator import MetricEvaluator
    stanford_radiogenomics = PATCH_DATASETS["StanfordRadiogenomicsPatchDataset"](root_dir=None,
                                                                                 transform=sitk2tensor,
                                                                                 split='test')
    print("length of stanford_radiogenomics dataset",
          len(stanford_radiogenomics))
    lndb_dl = DataLoader(dataset=stanford_radiogenomics,
                         batch_size=36,
                         shuffle=False,
                         drop_last=False,
                         num_workers=4,
                         pin_memory=True)
    re = ReconEvaluator(vis_dir=osp.join(os.getcwd(), "evaluations/results/"),
                        log_name='VAE3D32AUG',
                        version=version)
    re(dataloader=lndb_dl)
    pass


def test_metrics_for_stf_dataset():
    from datasets import PATCH_DATASETS
    import os.path as osp
    from datasets.utils import sitk2tensor
    from evaluations.evaluator import ReconEvaluator
    import numpy as np
    from evaluations.evaluator import MetricEvaluator
    stanford_radiogenomics = PATCH_DATASETS["StanfordRadiogenomicsPatchDataset"](root_dir=None,
                                                                                 transform=sitk2tensor,
                                                                                 split='test')
    print("length of stanford_radiogenomics dataset",
          len(stanford_radiogenomics))
    lndb_dl = DataLoader(dataset=stanford_radiogenomics,
                         batch_size=36,
                         shuffle=False,
                         drop_last=False,
                         num_workers=4,
                         pin_memory=True)
    me = MetricEvaluator(metrics=["SSIM", "MSE", "PSNR"],
                         log_name='VAE3D32AUG',
                         version=60)
    metrics_dict = me.calc_metrics(dataloader=lndb_dl)
    for k, v in metrics_dict.items():
        print(f"{k}: mean value = {np.mean(v)}")
    pass


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
                         batch_size=39,
                         shuffle=False,
                         drop_last=False,
                         num_workers=4,
                         pin_memory=True)
    me = MetricEvaluator(metrics=["SSIM", "MSE", "PSNR"],
                         log_name='VAE3D32AUG',
                         version=60)
    metrics_dict = me.calc_metrics(dataloader=lndb_dl)
    for k, v in metrics_dict.items():
        print(f"{k}: mean value = {np.mean(v)}")
    pass


if __name__ == "__main__":
    for v in [49, 51, 53, 57, 58, 59, 60]:
        test_recon_images_for_stf_dataset(version=v)
    # test_metrics_calculation_for_lidc_dataset()
    # test_metrics_for_stf_dataset()
    # test_metrics_calculation_for_lndb_dataset()
