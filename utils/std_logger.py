# Copyright Yongcheng He, 2024.10.18

import logging
import time
import os

# Create a custom logger with a unique name
logger = logging.getLogger('amphgt_logger')
logger.setLevel(logging.DEBUG)  # Set the default logging level
logger.propagate = False  # Prevent messages from propagating to ancestor loggers

# Flag to prevent re-initialization
_logger_initialized = False

def initialize_logger(log_file_path):
    global _logger_initialized

    if not _logger_initialized:
        # File handler
        file_handler = logging.FileHandler(os.path.join(log_file_path, "workflow.log"))
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter(
            '[%(asctime)s]-[%(filename)s]-[%(funcName)s]-[%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to our custom logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        _logger_initialized = True

# Wrapper functions to set the correct stack level
def debug(msg, *args, st=0, **kwargs):
    if st > 0:
        time.sleep(st)
    logger.debug(msg, *args, stacklevel=2, **kwargs)

def info(msg, *args, st=0, **kwargs):
    if st > 0:
        time.sleep(st)
    logger.info(msg, *args, stacklevel=2, **kwargs)

def warning(msg, *args, st=0, **kwargs):
    if st > 0:
        time.sleep(st)
    logger.warning(msg, *args, stacklevel=2, **kwargs)

def error(msg, *args, st=0, **kwargs):
    if st > 0:
        time.sleep(st)
    logger.error(msg, *args, stacklevel=2, **kwargs)
    raise Exception(msg)

def critical(msg, *args, st=0, **kwargs):
    if st > 0:
        time.sleep(st)
    logger.critical(msg, *args, stacklevel=2, **kwargs)
    raise Exception(msg)

def set_log_level(level):
    """
    Sets the logging level for the logger.
    """
    logger.setLevel(level)