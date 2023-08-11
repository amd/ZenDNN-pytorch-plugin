#******************************************************************************
# Copyright (c) 2023 Advanced Micro Devices, Inc.
# All rights reserved.
#******************************************************************************

# install requirements
pip install -r requirements.txt

# export PLUGIN_DIR
export PT_PLUGIN_DIR=$(pwd)
echo "PT_PLUGIN_DIR: $PT_PLUGIN_DIR"

# export ZENDNN_PARENT_FOLDER
cd ..
export ZENDNN_PARENT_FOLDER=$(pwd)
echo "ZENDNN_PARENT_FOLDER: $ZENDNN_PARENT_FOLDER"

cd $PT_PLUGIN_DIR

# Env variables set to copy ZenDNN/BLIS from local
# After ZenDNN4.1 release ZENDNN_PT_USE_LOCAL_ZENDNN should be set to 0
export ZENDNN_PT_USE_LOCAL_ZENDNN=1
export ZENDNN_PT_USE_LOCAL_BLIS=0

# to clean CMake intermediate files
mkdir -p build && cd build
rm -rf ./*

# to build again using cmake
cmake ..
make -j

# to build the plugin
cd ..
python setup.py bdist_wheel
WHL_FILE=$(find dist -name *.whl -type f -printf "%T@ %p\n" | sort -n | cut -d' ' -f 2- | tail -n 1)
pip install --force-reinstall $WHL_FILE
# python -m pip install --force-reinstall --no-cache-dir $WHL_FILE

# to check the config of torch_zendnn_plugin
python -c 'import torch; import torch_zendnn_plugin as zentorch; print(*zentorch._C.__config__.split("\n"), sep="\n")'

# to test the plugin is successfully built and installed
echo "Running PT PLUGIN Tests:"
python test/test_zendnn.py