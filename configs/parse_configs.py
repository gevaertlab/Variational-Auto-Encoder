""" to process config files to the training/evaluation module """
import argparse
import os
from functools import partial

import yaml
from utils.funcs import check_dict, edit_dict_value, iterate_nested_dict

from .config_vars import BASE_DIR
from datasets import PATCH_DATASETS


def _get_file_path(filename):
    file_path = os.path.join(BASE_DIR, 'configs', filename if filename.endswith(
        ".yaml") else filename + '.yaml')
    return file_path


def parse_config():
    # arg parser
    parser = argparse.ArgumentParser(description='Train VAE models')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help='config file name in /configs folder',
                        default='exp_new/vae32aug_exp')
    parser.add_argument('--note', "-N",
                        dest="note",
                        help='any note for training, will be saved in config file',
                        default="")
    parser.add_argument("--info", "-I",
                        dest="info",
                        help="flag to output information but not train",
                        action="store_true")
    args = parser.parse_args()
    config = process_config(args.filename)
    config['note'] = args.note
    if args.info:
        config['info'] = True
    return config


def load_config(filename):
    file_path = _get_file_path(filename)
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def process_config(filename):
    """
    this function make sure that the config format and items are usable and
    universally applicable by referring to 'template.yaml' config file
    """
    config = load_config(filename)
    ref_config = load_config('template')
    # modify some config params
    file_path = _get_file_path(filename)
    config['file_name'] = os.path.basename(file_path)
    config = compare_config(config, ref_config)

    # init logging directory
    logging_path = os.path.join(BASE_DIR,
                                config['logging_params']['save_dir'],
                                config['logging_params']['name'])
    if not os.path.exists(logging_path):
        print("creating logging directory")
        os.mkdir(logging_path)
    return config


def compare_config(config, ref_config):
    check_config_item_partial = partial(check_config_item_iter, config=config)
    iterate_nested_dict(ref_config,
                        check_config_item_partial)
    return config


def check_config_item_iter(ref_config,
                           keychain,
                           ref_value,
                           config):
    check_config_item(config,
                      keychain,
                      ref_value)
    pass


def check_config_item(config,
                      keychain,
                      ref_value):
    """
    compare an item in config and referenced version:
    optional -> none in config -> add
    optional -> sth in config -> skip
    required -> none in config -> return error
    required -> sth in config -> skip
    """
    # decide what to do by checking config
    if not check_dict(config, keychain):
        if ref_value.startswith('optional'):
            default_value = extract_ref_value(ref_value)
            edit_dict_value(config,
                            keychain,
                            default_value)
        else:
            raise ValueError(f"item {keychain} is required")
    pass


def extract_ref_value(ref_value):
    assert ref_value.startswith('optional ')
    value = ref_value.split(' ')[-1]
    if value.isnumeric():
        value = float(value)
        if value - int(value) == 0:
            value = int(value)
    return value
