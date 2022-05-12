import sys
import argparse
import numpy as np
from torch.utils.data.dataloader import DataLoader

sys.path.insert(1, "/labs/gevaertlab/users/yyhhli/code/vae/")


def reconstruction_eval():

    parser = argparse.ArgumentParser(
        description='Reconstruction metrics calc for diff datasets')
    parser.add_argument("--version", "-v",
                        default=56)
    parser.add_argument("--log_name", "-ln", default="VAE3DPERCEPTUAL")
    parser.add_argument("--ds_name", "-d", default="lidc")
    parser.add_argument("--base_model_name", "-bm", default="VAE3DPerceptual")
    args = parser.parse_args()

    from datasets import PATCH_DATASETS
    from datasets.utils import sitk2tensor
    from evaluations.evaluator import MetricEvaluator
    ds_dict = {"stf": "StanfordRadiogenomicsPatchDataset",
               "lidc": "LIDCPatchAugDataset",
               "lndb": "LNDbPatch32Dataset"}

    patch_ds = PATCH_DATASETS[ds_dict[args.ds_name]](root_dir=None,
                                                     transform=sitk2tensor,
                                                     split='val')
    print(f"length of {ds_dict[args.ds_name]} dataset",
          len(patch_ds))
    lndb_dl = DataLoader(dataset=patch_ds,
                         batch_size=36,
                         shuffle=False,
                         drop_last=False,
                         num_workers=4,
                         pin_memory=True)
    me = MetricEvaluator(metrics=['SSIM', 'MSE', 'PSNR'],
                         log_name=args.log_name,
                         version=args.version,
                         base_model_name=args.base_model_name)
    metrics_dict = me.calc_metrics(dataloader=lndb_dl)
    result_dict = {}
    for k, v in metrics_dict.items():
        print(f"{k}: mean value = {np.mean(v)}")
        result_dict[k] = np.mean(v)
    return result_dict


if __name__ == "__main__":
    result_dict = reconstruction_eval()
