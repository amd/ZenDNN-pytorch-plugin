#******************************************************************************
# Copyright (c) 2023 Advanced Micro Devices, Inc.
# All rights reserved.
#******************************************************************************

# install requirements
pip install -r requirements.txt

# build the plugin

# Alternative frontend to build wheel file
# Invokes the required version of setuptools, instead of directly invoking 
# setup.py 
#   python -m build --wheel --no-isolation

python setup.py bdist_wheel
WHL_FILE=$(find dist -name *.whl -type f -printf "%T@ %p\n" | sort -n | cut -d' ' -f 2- | tail -n 1)
pip install --force-reinstall $WHL_FILE

# to check the config of torch_zendnn_plugin
python -c 'import torch; import torch_zendnn_plugin as zentorch; print(*zentorch._C.__config__.split("\n"), sep="\n")'

# to test the plugin is successfully built and installed
echo "Running PT PLUGIN Tests:"
python test/test_zendnn.py