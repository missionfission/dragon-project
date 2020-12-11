import logging


def create_logger(
    log_file=None, log_file_level=logging.INFO, stats_file="logs/stats.txt"
):

    logger = logging.getLogger()
    if logger.handlers:
        logger.handlers = []

    file_handler = logging.FileHandler(stats_file, mode="w")
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    debug_file_handler = logging.FileHandler("logs/debug.txt", mode="w")
    debug_file_handler.setLevel(logging.DEBUG)
    logger.addHandler(debug_file_handler)

    # file_handler.setFormatter(log_format)

    return logger
