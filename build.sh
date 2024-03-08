#******************************************************************************
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
# All rights reserved.
#******************************************************************************

# exit when any command fails
set -e

# install requirements
pip install -r requirements.txt

# build the plugin

# Alternative frontend to build wheel file
# Invokes the required version of setuptools, instead of directly invoking
# setup.py
#   python -m build --wheel --no-isolation

# to build in debug mode 'export DEBUG=1'
python setup.py bdist_wheel

WHL_FILE=$(find dist -name *.whl -type f -printf "%T@ %p\n" | sort -n | cut -d' ' -f 2- | tail -n 1)
if [ "$SCRIPT_TYPE" == many_linux ];
then
    mv -f "$WHL_FILE" "${WHL_FILE%linux_x86_64.*}manylinux2014_x86_64.${WHL_FILE##*.}"
    WHL_FILE=$(find dist -name *.whl -type f -printf "%T@ %p\n" | sort -n | cut -d' ' -f 2- | tail -n 1)
fi
pip uninstall -y zentorch && pip install $WHL_FILE

# to check the config of zentorch
python -c 'import torch; import zentorch; print(*zentorch.__config__.split("\n"), sep="\n")'


# to test the plugin is successfully built and installed
echo "Running PT PLUGIN Tests:"
python test/test_zentorch.py