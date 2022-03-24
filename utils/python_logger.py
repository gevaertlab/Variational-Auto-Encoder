

import logging
import datetime as dt
import os
import os.path as osp


def get_logger(cls_name=None, create_file=True):
    logger = logging.getLogger(cls_name)

    if logger.handlers:
        logger.handlers = []

    logger.setLevel(level=logging.INFO)

    if cls_name:
        formatter = logging.Formatter(
            "[%(asctime)8s | %(module)s:%(name)10s] %(message)s",
            datefmt="%m-%d %H:%M:%S")
    else:
        formatter = logging.Formatter(
            "[%(asctime)8s | %(module)s:%(funcName)10s] %(message)s",
            datefmt="%m-%d %H:%M:%S")

    if create_file:
        file_path = create_logging_file()
        file_handler = logging.FileHandler(file_path, "w")
        file_handler.setLevel(level=logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.propagate = False  # won't hit parent logger
    return logger


def create_logging_file(root_dir="logging_files"):
    if not osp.exists(root_dir):
        os.mkdir(root_dir)
    date = dt.date.today().strftime("%m.%d")
    if not osp.exists(osp.join(root_dir, date)):
        os.mkdir(osp.join(root_dir, date))
    return osp.join(root_dir, date, "file.log")
