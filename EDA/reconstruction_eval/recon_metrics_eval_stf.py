import os
import os.path as osp
import sys
import argparse
import numpy as np
from torch.utils.data.dataloader import DataLoader

sys.path.insert(1, "/labs/gevaertlab/users/yyhhli/code/vae/")


def reconstruction_eval():

    parser = argparse.ArgumentParser(
        description='Reconstruction metrics calc for stf dataset')
    parser.add_argument("--version", "-v",
                        help="version of the dataset, if not specified, will use the latest version",
                        type=str,
                        default=70)
    parser.add_argument(
        "--log_name", "-ln", default="VAE3D32AUG")
    args = parser.parse_args()

    from datasets import PATCH_DATASETS
    from datasets.utils import sitk2tensor
    from evaluations.evaluator import MetricEvaluator
    stanford_radiogenomics = PATCH_DATASETS["StanfordRadiogenomicsPatchDataset"](root_dir=None,
                                                                                 transform=sitk2tensor,
                                                                                 split='test')
    print("length of stanford_radiogenomics dataset",
          len(stanford_radiogenomics))
    dataloader = DataLoader(dataset=stanford_radiogenomics,
                            batch_size=36,
                            shuffle=False,
                            drop_last=False,
                            num_workers=4,
                            pin_memory=True)
    me = MetricEvaluator(metrics=['SSIM', 'MSE', 'PSNR'],
                         log_name=args.log_name,
                         version=args.version,
                         base_model_name='VAE3D',
                         verbose=True)
    metrics_dict = me.calc_metrics(dataloader=dataloader)
    result_dict = {}
    for k, v in metrics_dict.items():
        print(f"{k}: mean value = {np.mean(v)}; std = {np.std(v)}")
        result_dict[k] = {'mean': float(np.mean(v)),
                          'std': float(np.std(v)),
                          'values': [float(value) for value in v]}
    # save result to results/ folder as a json file
    log_name = args.log_name.replace("/", "-")
    dname = os.path.dirname(os.path.abspath(__file__))
    with open(osp.join(dname, f"results/{log_name}_version-{me.version}.json"), "w") as f:
        import json
        json.dump(result_dict, f)
    return result_dict


if __name__ == "__main__":
    result_dict = reconstruction_eval()
