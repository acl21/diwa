#!/bin/bash

# Download, Unzip, and Remove zip
if [ "$1" = "calvin" ]
then
    cd DIWA_DATA_DIR/ && mkdir calvin && cd calvin
    echo "Downloading calvin task_D_D ..."
    wget http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip
    unzip task_D_D.zip && rm task_D_D.zip
    mv task_A_A task_D_D
    echo "saved folder: task_D_D"
elif [ "$1" = "real" ]
then
    cd DIWA_DATA_DIR/ && mkdir real && cd real
    echo "Downloading real task ..."
    wget http://diwa.cs.uni-freiburg.de/download/data/real_play.zip
    unzip real_play.zip && rm real_play.zip
    echo "saved folder: real_play"
else
    echo "Failed: Usage download_data.sh calvin/real | XXX "
    exit 1
fi