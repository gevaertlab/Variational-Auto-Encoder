"""
call preprocess module 
"""

from patch_extraction.patch_extract import PatchExtract
from patch_extraction import registration, aug_params
import argparse
from configs.config_vars import DS_ROOT_DIR
import os.path as osp
from datasets import CT_DATASETS


def param_parser():
    parser = argparse.ArgumentParser(description='preprocess datasets')
    parser.add_argument('--dataset',  '-r',
                        dest="dataset",
                        help='name of the dataset, defined in CT_DATASETS',
                        default='LIDCDataset')  # HACK: should be deprecated
    parser.add_argument('--aug_param',  '-a',
                        dest="aug_param",
                        help='augmentation parameters (filename) for '
                        'aumgentation, should be in aug_param folder',
                        default='rotate')
    parser.add_argument('--save_dir',  '-S',
                        dest="save_dir",
                        help="save directory of converted patches",
                        default='TCIA_LIDC/LIDC-patch-32_aug')
    parser.add_argument('--vis_dir',  '-V',
                        dest="vis_dir",
                        help="visualization directory of converted patches",
                        default='TCIA_LIDC/LIDC-visualization-32_aug')
    parser.add_argument('--size',  '-s',
                        dest="size",
                        help="size of the patches",
                        default='32')
    parser.add_argument('--multi',  '-M',
                        dest="multi",
                        action='store_true',
                        help="whether to use multiprocessing")
    parser.add_argument('--debug',
                        dest="debug",
                        action='store_true',
                        help="debug flag for preprocessing, "
                             "will output examples of extracted patches for debugging")
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
    ds = CT_DATASETS[args.dataset](
        params={"save_path": "/labs/gevaertlab/data/lung cancer/TCIA_LIDC/lidc_info_dict.json"})
    # ds_register = getattr(registration, "LidcReg")()
    # ds_register.register()
    # dataset_params = ds_register.info_dict
    ds_params = ds.generate_ds_params()
    # augmentation parameters
    aug_param = getattr(aug_params, args.aug_param).augmentation_params
    # extract patches
    patch_extract = PatchExtract(patch_size=args.size,
                                 ds_params=ds_params,
                                 augmentation_params=aug_param,
                                 debug=args.debug)
    patch_extract.load_extract_ds(save_dir=args.save_dir,
                                  multi=args.multi)
    # visualization
    if args.debug:
        patch_extract.vis_ds(dataset_dir=args.save_dir,
                             vis_dir=args.vis_dir)
