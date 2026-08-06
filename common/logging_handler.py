import logging
import tempfile
import os


DEFAULT_LOG_FORMAT = (
    "%(asctime)s %(levelname)-8s %(message)-80s "
    "%(filename)s:[%(lineno)-6s - %(funcName)-20s()]"
)


def summarize_exception(error: Exception) -> str:
    """Return a compact, single-line exception chain for operational logs."""
    summaries = []
    current_error = error
    visited = set()

    while current_error is not None and id(current_error) not in visited:
        visited.add(id(current_error))
        error_type = type(current_error).__name__
        error_message = " ".join(str(current_error).split())
        summaries.append(
            f"{error_type}: {error_message}" if error_message else error_type
        )
        current_error = current_error.__cause__ or current_error.__context__

    return " | caused by: ".join(summaries)


def log_workflow_exception(workflow_name: str, error: Exception) -> None:
    """Log a traceback followed by the final one-line ERROR consumed by HAT."""
    logging.error(
        " ==> ERROR! %s failed. Full traceback follows.",
        workflow_name,
        exc_info=(type(error), error, error.__traceback__),
        stacklevel=2,
    )
    logging.error(
        " ==> ERROR! %s failed. Last error summary: %s",
        workflow_name,
        summarize_exception(error),
        stacklevel=2,
    )


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
    logger_folder: str = None,
    logger_file: str = None,
    logger_format: str = DEFAULT_LOG_FORMAT,
    logger_level: int = logging.INFO,
    logger_name: str = "logger"
    ) -> str:
    """
    Set the logging stream.

    :param logger_folder: Folder to save the log file.
    :param logger_file: Name of the log file.
    :param logger_format: Format of the log messages.
    :param logger_level: Logging level.
    :param logger_name: Name of the logger.
    :return: Path to the log file.
    """
    reset_logging_stream(logger_name=logger_name)

    if logger_format is None:
        logger_format = DEFAULT_LOG_FORMAT
    if logger_folder is None:
        logger_folder = tempfile.gettempdir()
    if logger_file is None:
        logger_file = f"{logger_name}.log"

    logger_path = os.path.join(logger_folder, logger_file)
    setup_logging(logger_path, logger_format, logger_level, logger_name)
    return logger_path


def setup_logging(logger_path: str, logger_format: str, logger_level: int, logger_name: str) -> None:
    """
    Helper function to set up logging.

    :param logger_path: Path to the log file.
    :param logger_format: Format of the log messages.
    :param logger_level: Logging level.
    :param logger_name: Name of the logger.
    """
    logger_path = os.path.abspath(logger_path)
    os.makedirs(os.path.dirname(logger_path), exist_ok=True)

    logger_formatter = logging.Formatter(logger_format)
    file_handler = logging.FileHandler(logger_path, mode="w")
    stream_handler = logging.StreamHandler()

    for handler in (file_handler, stream_handler):
        handler.setLevel(logger_level)
        handler.setFormatter(logger_formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logger_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    named_logger = logging.getLogger(logger_name)
    named_logger.setLevel(logger_level)
    named_logger.propagate = True
