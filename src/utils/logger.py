import logging


def create_logger(log_file=None, log_file_level=logging.NOTSET):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers = []

    if log_file and log_file != "":
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        logger.addHandler(file_handler)
        # file_handler.setFormatter(log_format)

    return logger
