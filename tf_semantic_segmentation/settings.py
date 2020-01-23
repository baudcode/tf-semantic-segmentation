import logging


def get_logger(name='tf-semantic-segmentation'):

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    _formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()

    # output of subprocess call will not be printed into the console if using logging.INFO
    # this prohibits printing of subprocess call output twice
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(_formatter)
    if len(logger.handlers) == 0:
        logger.addHandler(console_handler)
    return logger


logger = get_logger()
