#!/bin/bash

set -e

if grep -q docker /proc/1/cgroup; then 
    echo "Cannot run inside Docker environment because of Agisoft Metashape Pro license. Run on host..."
    exit 
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 [yolo_uncertainty] [mcd_early_occ_cls_iou] [mcd_early_occ_cls2_iou] [mcd_early_cls_iou] [mcd_early_occ_cls] 
        [mcd_early_occ_iou] [mcd_early_occ] [mcd_early_cls] [mcd_early_cls2] [mcd_early_iou] [mcd_early_yolo_mean] 
        [mcd_late_occ_cls_iou] [mcd_late_occ_cls2_iou] [mcd_late_cls_iou] [mcd_late_occ_cls] [mcd_late_occ_iou] [mcd_late_occ] 
        [mcd_late_cls] [mcd_late_cls2] [mcd_late_iou] [mcd_late_yolo_mean]"
    exit 1
fi

contains_option() {
    local target="$1"
    shift
    
    for arg in "$@"; do
        if [ "$arg" == "$target" ]; then
            return 0 
        fi
    done
    
    return 1 
}

DATASET_PATH="/media/rick/DATA_RICK/adaptive-planning/"

if contains_option "gt" "$@"; then
    for dataset in "clustered_1" "clustered_2" "clustered_3" "clustered_4" "uniform_1" "uniform_2" "uniform_3" "uniform_4"; do
        for folder in "${DATASET_PATH}/"*/; do
            if [[ "${folder}" == *${dataset}/ ]]; then
                for altitude in "12m" "24m" "32m"; do
                    image_gt_gps ${folder}/${altitude}.psx ${folder}/plants_${dataset}_RDNAPTRANS2008.csv detection_evaluation_results/${dataset}_${altitude}_gt.kml
                done    
                continue
            fi
        done
    done
fi

process_with_uncertainty_method() {
    local uncertainty_method="$1"

    # for dp in "0.25" "0.50" "0.75"; do
    for dp in "0.25" "0.50"; do
        for dataset in "clustered_1" "clustered_2" "clustered_3" "clustered_4" "uniform_1" "uniform_2" "uniform_3" "uniform_4"; do
            for folder in "${DATASET_PATH}/"*/; do
                if [[ "${folder}" == *${dataset}/ ]]; then
                    for altitude in "12m" "24m" "32m"; do                    
                        image_folders=""
                        for f in "${folder}/"*/; do
                            if [[ "$(basename "${f}")" == DJI* && ("${f}" == *"${altitude}/" || "${f}" == *"${altitude%m}/") ]]; then
                                image_folders+="${f} "
                            fi
                        done
                        image_detect_gps adaptive_planner/best_n.pt ${folder}/${altitude}.psx detection_evaluation_results/${dp}_${uncertainty_method}/${dataset}_${altitude}_${uncertainty_method}.kml ${image_folders} --uncertainty_method ${uncertainty_method} --dropout_probability ${dp}
                    done    
                    continue
                fi
            done
        done
    done
}

declare -A method_mapping
method_mapping=(
    ["yolo_uncertainty"]="yolo_uncertainty"
    ["mcd_early_occ_cls_iou"]="mcd_early_occ_cls_iou"
    ["mcd_early_occ_cls2_iou"]="mcd_early_occ_cls2_iou"
    ["mcd_early_cls_iou"]="mcd_early_cls_iou"
    ["mcd_early_occ_cls"]="mcd_early_occ_cls"
    ["mcd_early_occ_cls2"]="mcd_early_occ_cls2"
    ["mcd_early_occ_iou"]="mcd_early_occ_iou"
    ["mcd_early_occ"]="mcd_early_occ"
    ["mcd_early_cls"]="mcd_early_cls"
    ["mcd_early_iou"]="mcd_early_iou"
    ["mcd_early_yolo_mean"]="mcd_early_yolo_mean"
    ["mcd_late_occ_cls_iou"]="mcd_late_occ_cls_iou"
    ["mcd_late_occ_cls2_iou"]="mcd_late_occ_cls2_iou"
    ["mcd_late_cls_iou"]="mcd_late_cls_iou"
    ["mcd_late_occ_cls"]="mcd_late_occ_cls"
    ["mcd_late_occ_iou"]="mcd_late_occ_iou"
    ["mcd_late_occ"]="mcd_late_occ"
    ["mcd_late_cls"]="mcd_late_cls"
    ["mcd_late_cls2"]="mcd_late_cls2"
    ["mcd_late_iou"]="mcd_late_iou"
    ["mcd_late_yolo_mean"]="mcd_late_yolo_mean"
)

for key in "${!method_mapping[@]}"; do
    if contains_option "$key" "$@"; then
        process_with_uncertainty_method "${method_mapping[$key]}"
    fi
done