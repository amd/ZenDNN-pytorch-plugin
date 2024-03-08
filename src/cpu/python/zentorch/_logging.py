# ******************************************************************************
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import os
import logging


def get_logger(__name__):
    """
    get_logger:
    takes in the filename and,
    returns a logger based on logging.conf file
    """
    # define a message format
    FORMAT = "[%(levelname)s %(name)s - %(funcName)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=FORMAT)
    # make a logger for this file
    logger = logging.getLogger(__name__)
    # check if user has set some logging level
    if os.environ.get("ZENTORCH_PY_LOG_LEVEL") is not None:
        logger.setLevel(os.environ.get("ZENTORCH_PY_LOG_LEVEL"))

    return logger
