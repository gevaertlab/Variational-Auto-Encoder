

import logging


def get_logger(module, cls_name, create_file=False):
    logger = logging.getLogger(module)
    logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter(
        f"[%(asctime)10s | {module}:{cls_name:<10s}] %(message)s")

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
