"""
call preprocess module 
"""

from patch_extraction.patch_extract import PatchExtract
from patch_extraction.aug_params import AUG_PARAMS
import argparse
from configs.config_vars import DS_ROOT_DIR
import os.path as osp
from datasets import CT_DATASETS


def param_parser():
    parser = argparse.ArgumentParser(description='preprocess datasets')
    parser.add_argument('--dataset',  '-r',
                        dest="dataset",
                        help='name of the dataset, defined in CT_DATASETS',
                        default='LDNbDataset')
    parser.add_argument('--aug_param',  '-a',
                        dest="aug_param",
                        help='augmentation parameters (filename) for '
                        'aumgentation, should be in aug_param folder',
                        default='version1')  # in `aug_params`` rotate, shift or version1
    parser.add_argument('--save_dir',  '-S',
                        dest="save_dir",
                        help="save directory of converted patches",
                        default='LNDb/LNDb-patch32-aug')
    parser.add_argument('--vis_dir',  '-V',
                        dest="vis_dir",
                        help="visualization directory of converted patches",
                        default='')  # NOTE: debug e.g. StanfordRadiogenomics/patch-visualization-32-aug
    parser.add_argument('--size',  '-s',
                        dest="size",
                        help="size of the patches",
                        default='32')
    parser.add_argument('--multi',  '-M',
                        dest="multi",
                        action='store_true',
                        help="whether to use multiprocessing")
    parser.add_argument('--overwrite',
                        dest="overwrite",
                        help="whether to overwrite previously processed set",
                        action='store_true',)
    args = parser.parse_args()
    args.size = tuple([int(args.size)] * 3)
    if not args.save_dir.startswith('/'):
        args.save_dir = osp.join(DS_ROOT_DIR, args.save_dir)
    if not args.vis_dir.startswith('/'):
        args.vis_dir = osp.join(DS_ROOT_DIR, args.vis_dir)
    return args


if __name__ == "__main__":
    args = param_parser()
    # dataset registration
    ds = CT_DATASETS[args.dataset]()
    # augmentation parameters
    aug_param = AUG_PARAMS[args.aug_param]  # defined in aug_params
    # extract patches
    patch_extract = PatchExtract(patch_size=args.size,
                                 dataset=ds,
                                 augmentation_params=aug_param,
                                 debug=bool(args.vis_dir))
    patch_extract.load_extract_ds(save_dir=args.save_dir,
                                  multi=args.multi,
                                  overwrite=args.overwrite)
    # visualization
    if args.vis_dir:
        patch_extract.vis_ds(dataset_dir=args.save_dir,
                             vis_dir=args.vis_dir)
