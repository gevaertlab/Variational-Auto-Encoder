import sys
import argparse
import numpy as np
from torch.utils.data.dataloader import DataLoader

sys.path.insert(1, "/labs/gevaertlab/users/yyhhli/code/vae/")


def reconstruction_eval():

    parser = argparse.ArgumentParser(
        description='Reconstruction metrics calc for lndb dataset')
    parser.add_argument("--version", "-v",
                        default=60)
    parser.add_argument("--log_name", "-ln", default="VAE3D32AUG")
    args = parser.parse_args()

    from datasets import PATCH_DATASETS
    from datasets.utils import sitk2tensor
    from evaluations.evaluator import MetricEvaluator
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
    me = MetricEvaluator(metrics=['SSIM', 'MSE', 'PSNR'],
                         log_name=args.log_name,
                         version=args.version,
                         base_model_name='VAE3D')
    metrics_dict = me.calc_metrics(dataloader=lndb_dl)
    result_dict = {}
    for k, v in metrics_dict.items():
        print(f"{k}: mean value = {np.mean(v)}")
        result_dict[k] = np.mean(v)
    return result_dict


if __name__ == "__main__":
    result_dict = reconstruction_eval()
