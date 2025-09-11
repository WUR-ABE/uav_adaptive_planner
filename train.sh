#!/bin/bash

set -e

yolo settings wandb=True
train 2048n_more_blur \
    data=${DATA_HOME}/trainings_dataset/data.yml \
    project=paper_4 \
    model=yolo11n.pt \
    epochs=250 \
    imgsz=2048 \
    save=True \
    batch=-1 \
    flipud=0.5 \
    perspective=0.00005
