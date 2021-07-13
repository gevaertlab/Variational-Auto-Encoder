""" 
call preprocess module 
"""

from patch_extraction.patch_extract import PatchExtract
from patch_extraction import registration, aug_params
import argparse
from configs.config_vars import DS_ROOT_DIR
import os.path as osp


def param_parser():
    parser = argparse.ArgumentParser(description='preprocess datasets')
    parser.add_argument('--register',  '-r',
                        dest="register",
                        help='register name of the dataset, in registration folder, REGISTERS',
                        default='LidcReg')
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
    parser.add_argument('--not_vis',  '-v',
                        dest="visualize",
                        help="whether visualize each patch")
    parser.add_argument('--size',  '-s',
                        dest="size",
                        help="size of the patches",
                        default='32')
    parser.add_argument('--not_multi',  '-M',
                        dest="not_multi",
                        help="do not use multiprocessing")
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
    ds_register = getattr(registration, args.register)()
    ds_register.register()
    ds_register.print_info()
    dataset_params = ds_register.info_dict
    # augmentation parameters
    aug_param = getattr(aug_params, args.aug_param).augmentation_params
    # extract patches
    patch_extract = PatchExtract(patch_size=args.size,
                                 ds_params=dataset_params,
                                 augmentation_params=aug_param)
    patch_extract.extract_ds(save_dir=args.save_dir,
                             multi=(not args.not_multi))
    # visualization
    if not args.not_vis:
        patch_extract.vis_ds(dataset_dir=args.save_dir,
                             vis_dir=args.vis_dir)
