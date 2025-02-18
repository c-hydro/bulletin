import logging
import os
import sys
from copy import deepcopy

def reset_logging_stream(logger_name: str) -> None:
    """
    Reset the logging stream.

    :param logger_name: Name of the logger.
    """
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    loggers.append(logging.getLogger())
    for logger in loggers:
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(logging.NOTSET)
        logger.propagate = True

def set_logging_stream(
    logger_folder: str,
    logger_file: str,
    logger_format: str = "%(asctime)s %(levelname)-8s %(message)-80s %(filename)s:[%(lineno)-6s - %(funcName)-20s()]",
    logger_level: int = logging.INFO,
    logger_name: str = "logger"
    ) -> None:
    """
    Set the logging stream.

    :param logger_folder: Folder to save the log file.
    :param logger_file: Name of the log file.
    :param logger_format: Format of the log messages.
    :param logger_level: Logging level.
    :param logger_name: Name of the logger.
    """
    reset_logging_stream(logger_name=logger_name)

    if logger_format is None:
        logger_format = deepcopy(logger_format)
    if logger_file is None:
        logger_file = deepcopy(logger_file)

    if logger_folder is not None:
        logger_path = os.path.join(logger_folder, logger_file)
    else:
        logger_path = deepcopy(logger_file)

    if os.path.exists(logger_path):
        os.remove(logger_path)

    logger_loc = os.path.split(logger_path)
    if logger_loc[0] == '':
        logger_folder_name, logger_file_name = os.path.dirname(os.path.abspath(sys.argv[0])), logger_loc[1]
    else:
        logger_folder_name, logger_file_name = logger_loc[0], logger_loc[1]

    os.makedirs(logger_folder_name, exist_ok=True)
    logger_path = os.path.join(logger_folder_name, logger_file_name)

    if os.path.exists(logger_path):
        os.remove(logger_path)

    logging.getLogger(logger_name)
    logging.root.setLevel(logger_level)
    logging.basicConfig(level=logger_level, format=logger_format, filename=logger_path, filemode='w')

    logger_handle_1 = logging.FileHandler(logger_path, 'w')
    logger_handle_2 = logging.StreamHandler()
    logger_handle_1.setLevel(logger_level)
    logger_handle_2.setLevel(logger_level)
    logger_formatter = logging.Formatter(logger_format)
    logger_handle_1.setFormatter(logger_formatter)
    logger_handle_2.setFormatter(logger_formatter)

    logging.getLogger('').addHandler(logger_handle_1)
    logging.getLogger('').addHandler(logger_handle_2)