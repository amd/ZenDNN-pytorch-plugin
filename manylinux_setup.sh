#******************************************************************************
# Copyright (c) 2023 Advanced Micro Devices, Inc.
# All rights reserved.
#******************************************************************************

#Moving to PARENT_FOLDER
cd ../
if [ $( docker ps -a | grep -w manylinux2014_zendnn_plugin | wc -l ) -gt 0 ]; then
    echo "Manylinux 2014 docker exists"
    if [ $( docker ps -af status=exited | grep -w manylinux2014_zendnn_plugin | wc -l ) -gt 0 ];
    then
        echo "Booting manylinux2014 docker"
        docker start manylinux2014_zendnn_plugin
    fi
    echo "Entering the docker image"
    docker exec -it manylinux2014_zendnn_plugin bash
else
    echo "Creating and entering a new manylinux2014 docker"
    docker run --name manylinux2014_zendnn_plugin -itd --privileged -v $(pwd):/home quay.io/pypa/manylinux2014_x86_64:2021-07-14-67a6e11
    docker start manylinux2014_zendnn_plugin
    docker exec -it manylinux2014_zendnn_plugin yum install -y sshpass openssh-clients wget python zip unzip ccache hwloc dmidecode
    docker exec -it manylinux2014_zendnn_plugin bash
fi
