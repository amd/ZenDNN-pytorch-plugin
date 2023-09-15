# ******************************************************************************
# Copyright (c) 2023 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import os
import logging
import logging.config


def get_logger(__name__):
    """
    get_logger:
    takes in the filename and,
    returns a logger based on logging.conf file
    """
    # use the logging configurations from a .conf file
    logging.config.fileConfig("src/cpu/python/torch_zendnn_plugin/logging.conf")
    # make a logger for this file
    logger = logging.getLogger(__name__)
    # check if user has set some logging level
    if os.environ.get("ZENTORCH_PY_LOG_LEVEL") is not None:
        logger.setLevel(os.environ.get("ZENTORCH_PY_LOG_LEVEL"))

    return logger
