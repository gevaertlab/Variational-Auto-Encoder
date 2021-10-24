

import logging


def get_logger(cls_name=None, create_file=False):
    logger = logging.getLogger(cls_name)
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
        file_handler = logging.FileHandler('file.log')
        file_handler.setLevel(level=logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    return logger
