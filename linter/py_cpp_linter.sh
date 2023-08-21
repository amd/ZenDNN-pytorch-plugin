# ******************************************************************************
# Copyright (c) 2023 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR
pip install --upgrade -r requirements.txt
cd ../

# Flake8
echo "********************************************************************************"
echo "            * Starting flake8 linting for python scripts... *"
echo "********************************************************************************"
flake8
echo "To re-format the above files (if any) with black, run the following command (files will not always be changed): "
tput bold
echo "flake8 --quiet | xargs black --verbose"
tput sgr0
echo "Completed py-linting!"
echo -e "********************************************************************************\n\n"

# clang-format
echo "********************************************************************************"
echo "          * Now executing clang-format checks for C++ files... *"
echo "********************************************************************************"
git clang-format --commit `git rev-list HEAD | tail -n 1` --diff
echo "If clang-format suggested some modifications, then you can use the following command to re-format: "
tput bold
echo "git clang-format -f"
tput sgr0
echo "CPP linting completed!"
echo "********************************************************************************"
