#!/bin/bash

set -e

if grep -q docker /proc/1/cgroup; then 
    DATASET_PATH=/home/abe/data/
else
    DATASET_PATH=/media/rick/DATA_RICK/adaptive-planning
fi

yolo settings wandb=True
train 2048n_more_blur \
    data=${DATASET_PATH}/trainings_dataset/data.yml \
    project=paper_4 \
    model=yolo11n.pt \
    epochs=250 \
    imgsz=2048 \
    save=True \
    batch=-1 \
    flipud=0.5 \
    perspective=0.00005
