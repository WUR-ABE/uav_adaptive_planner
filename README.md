# Adaptive path planning for UAVs

![flight-path](assets/adaptive_planner.gif "flight-path-example")
> **Adaptive path planning for efficient object search by UAVs in agricultural fields**\
> Rick van Essen, Eldert van Henten, Lammert Kooistra and Gert Kootstra
> Paper: https://arxiv.org/abs/2504.02473

## About
This repository contains the code beloning to the paper "Adaptive path planning for efficient object search by UAVs in agricultural fields". 

## Installation
Python 3.10 is needed with all dependencies listed in [`requirements.txt`](requirements.txt) and some additional dependencies. Install using:

```commandline
sudo apt-get install libimage-exiftool-perl
pip install -r .
```

Additionally, `Fields2Cover` is needed. See their [website](https://fields2cover.github.io/index.html) for installation instructions.

Alternatively, you can use the provided [dev container](.devcontainer).

## Run on custom orthomosaic dataset

### Dataset preperation
> **_NOTE:_**  Some tools require the Agisoft Metashape Python module (with valid license). See their [website](https://www.agisoft.com/downloads/installer/) for installation instructions.


1. Transform Topcon data files CRS from `RDNAPTRANS2008` to `WGS84`:

```commandline
transform_markers <<path_to_markers_file_rdnaptrans2008>> <<path_to_markers_file_WGS84>> epsg:4326
```
2. Create plot of marker locations:

```commandline
plot_markers <<path_to_markers_file>> --output_file <<path_to_output_file>>
```
3. Build orthomosaics for the 12m altitude images
4. Align images for the 24m and 32m altitude images
5. Select images for train and validation:

```commandline
create_detection_dataset <<path_to_markers_file>> <<path_to_dataset>> <<path_to_metashape_project>>
```
6. Refine the auto-generated labels using a labeling tool (for example `labelImg`):

```commandline
labelImg <<path_to_dataset>>/images/train <<path_to_dataset>>/classes.txt <<path_to_dataset>>/labels/train
labelImg <<path_to_dataset>>/images/val <<path_to_dataset>>/classes.txt <<path_to_dataset>>/labels/val
```

7. Mask out annotations that are not training and validation and create YOLO dataset:

```commandline
mask_detection_dataset <<path_to_dataset>>/data.yml <<path_to_train_markers_file>> <<path_to_validation_markers_file>> <<path_to_metashape_project>>
```

### Training
1. Adapt the paths in `train.sh` to refer to your custom dataset
2. Run training:

```commandline
./train.sh
```

The weights will be saved in [`training_results`](training_results/).

### Run adaptive path planner
1. Draw a field and boundary file to indicate the area of interest to the planner:
```commandline
draw_field --output_folder fields --name <<name of field>> --scheme_file <<path to orthomosiac scheme file (optional, as reference for drawing)>>
```
2. Run the adaptive path planner:
```commandline
adaptive_planner orthomosaic_sim <<path to field yaml file>> <<path to planner config file>> --name <<experiment name>>
```

## Citation
If you find this code usefull, please consider citing our paper:

```
@misc{essen2025,
    title = {Adaptive path planning for efficient object search by UAVs in agricultural fields},
    urldate = {2025-04-03},
    publisher = {arXiv},
    author = {van Essen, Rick and van Henten, Eldert and Kooistra, Lammert and Kootstra, Gert},
    month = apr,
    year = {2025},
    note = {arXiv:2504.02473 [cs]},
}
```

## Funding
This research is part of the research program SYNERGIA, funding was obtained from the Dutch Research Council (NWO grant 17626), IMEC-One Planet and other private parties.

<img src="assets/wageningen_logo.png" alt="wageningen university logo" height="50"> &nbsp;&nbsp;&nbsp; <img src="assets/synergia_logo_basis.png" alt="synergia logo" height="50">
