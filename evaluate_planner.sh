#!/bin/bash

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 [parameter_estimation_baseline] [parameter_estimation] [localization_uncertainty_baseline] [localization_uncertainty] [number_of_objects_baseline] [number_of_objects]"
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

# Parameter estimation baseline
if contains_option "parameter_estimation_baseline" "$@"; then
    for altitude in "12m" "24m" "36m" "48m"; do
        for dataset in "clustered_1" "clustered_2" "clustered_3" "clustered_4" "uniform_1" "uniform_2" "uniform_3" "uniform_4"; do
             if [ -d "planner_evaluation_results/parameter_estimation/baseline_${altitude}_${dataset}" ]; then
                echo "Skipping '${altitude}_${dataset}' because it already exists!"
                continue
            fi

            adaptive_planner orthomosaic_sim \
                fields/${dataset}.yaml \
                experiments/parameter_estimation/baseline_${altitude}.yaml \
                --output_folder planner_evaluation_results/parameter_estimation/baseline_${altitude}_${dataset}
        done
    done
fi

# Parameter estimation
if contains_option "parameter_estimation" "$@"; then
    for altitude in "12m" "24m" "36m" "48m"; do
        for accept_threshold in "0.4" "0.6" "0.8" "1.0"; do
            for reject_threshold in "0.05" "0.2" "0.4"; do 
                # Skip planner for situations that should not exist
                if [ "${accept_threshold}" = "${reject_threshold}" ]; then
                    continue
                fi

                for dataset in "clustered_1" "clustered_2" "clustered_3" "clustered_4" "uniform_1" "uniform_2" "uniform_3" "uniform_4"; do
                    if [ -d "planner_evaluation_results/parameter_estimation/alt_${altitude}_accept_${accept_threshold}_reject_${reject_threshold}_${dataset}" ]; then
                        echo "Skipping 'alt_${altitude}_accept_${accept_threshold}_reject_${reject_threshold}_${dataset}' because it already exists!"
                        continue
                    fi

                    adaptive_planner orthomosaic_sim \
                        fields/${dataset}.yaml \
                        experiments/parameter_estimation/alt_${altitude}_accept_${accept_threshold}_reject_${reject_threshold}.yaml \
                        --output_folder planner_evaluation_results/parameter_estimation/alt_${altitude}_accept_${accept_threshold}_reject_${reject_threshold}_${dataset} \
                        --base_flight_folder planner_evaluation_results/parameter_estimation/baseline_${altitude}_${dataset}
                done
            done
        done
    done
fi

# Position uncertainty
if contains_option "localization_uncertainty_baseline" "$@"; then
    for altitude in "12m" "24m" "36m" "48m"; do
        for category in "perfect" "good" "decent" "poor" "very_poor"; do
            for dataset in "clustered_1" "clustered_2" "clustered_3" "clustered_4"; do
                if [ -d "planner_evaluation_results/localization_uncertainty/baseline_${altitude}_${dataset}_${category}" ]; then
                    echo "Skipping 'baseline_${altitude}_${dataset}_${category}' because it already exists!"
                    continue
                fi

                adaptive_planner orthomosaic_sim \
                    fields/${dataset}.yaml \
                    experiments/localization_uncertainty/baseline_${altitude}_planner_clustered_${category}_config.yaml \
                    --output_folder planner_evaluation_results/localization_uncertainty/baseline_${altitude}_${dataset}_${category} \
                    --executor_config_file experiments/localization_uncertainty/${category}.yaml
            done
        done
    done

    for altitude in "12m" "24m" "36m" "48m"; do
        for category in "perfect" "good" "decent" "poor" "very_poor"; do
            for dataset in "uniform_1" "uniform_2" "uniform_3" "uniform_4"; do
                if [ -d "planner_evaluation_results/localization_uncertainty/baseline_${altitude}_${dataset}_${category}" ]; then
                    echo "Skipping 'baseline_${altitude}_${dataset}_${category}' because it already exists!"
                    continue
                fi

                adaptive_planner orthomosaic_sim \
                    fields/${dataset}.yaml \
                    experiments/localization_uncertainty/baseline_${altitude}_planner_uniform_${category}_config.yaml \
                    --output_folder planner_evaluation_results/localization_uncertainty/baseline_${altitude}_${dataset}_${category} \
                    --executor_config_file experiments/localization_uncertainty/${category}.yaml
            done
        done
    done
fi

if contains_option "localization_uncertainty" "$@"; then
    for altitude in "12m" "24m" "36m" "48m"; do
        for category in "perfect" "good" "decent" "poor" "very_poor"; do
            for dataset in "clustered_1" "clustered_2" "clustered_3" "clustered_4"; do
                if [ -d "planner_evaluation_results/localization_uncertainty/${altitude}_${dataset}_${category}" ]; then
                    echo "Skipping '${altitude}_${dataset}_${category}' because it already exists!"
                    continue
                fi

                adaptive_planner orthomosaic_sim \
                    fields/${dataset}.yaml \
                    experiments/localization_uncertainty/${altitude}_planner_clustered_${category}_config.yaml \
                    --output_folder planner_evaluation_results/localization_uncertainty/${altitude}_${dataset}_${category} \
                    --base_flight_folder planner_evaluation_results/localization_uncertainty/baseline_${altitude}_${dataset}_${category} \
                    --executor_config_file experiments/localization_uncertainty/${category}.yaml
            done
        done
    done

    for altitude in "12m" "24m" "36m" "48m"; do
        for category in "perfect" "good" "decent" "poor" "very_poor"; do
            for dataset in "uniform_1" "uniform_2" "uniform_3" "uniform_4"; do
                if [ -d "planner_evaluation_results/localization_uncertainty/${altitude}_${dataset}_${category}" ]; then
                    echo "Skipping '${altitude}_${dataset}_${category}' because it already exists!"
                    continue
                fi

                adaptive_planner orthomosaic_sim \
                    fields/${dataset}.yaml \
                    experiments/localization_uncertainty/${altitude}_planner_uniform_${category}_config.yaml \
                    --output_folder planner_evaluation_results/localization_uncertainty/${altitude}_${dataset}_${category} \
                    --base_flight_folder planner_evaluation_results/localization_uncertainty/baseline_${altitude}_${dataset}_${category} \
                    --executor_config_file experiments/localization_uncertainty/${category}.yaml
            done
        done
    done
fi

# Number of objects
if contains_option "number_of_objects_baseline" "$@"; then
    for number in "0" "20" "40" "60" "80" "100" "120" "140" "160" "180" "200"; do
        for alt in "12m" "36m"; do
            for dataset in "uniform_1" "uniform_2" "uniform_3" "uniform_4"; do
                if [ -d "planner_evaluation_results/number_of_objects/baseline_${alt}_${number}_${dataset}" ]; then
                    echo "Skipping 'baseline_${alt}_${number}_${dataset}' because it already exists!"
                    continue
                fi

                adaptive_planner orthomosaic_sim \
                    fields/${dataset}.yaml \
                    experiments/number_of_objects/baseline_${alt}_planner_parameters.yaml \
                    --output_folder planner_evaluation_results/number_of_objects/baseline_${alt}_${number}_${dataset} \
                    --executor_config_file experiments/number_of_objects/executor_${number}_objects_${dataset}.yaml
            done
        done
    done

    for number in "0" "20" "40" "60" "80" "100" "120" "140" "160" "180" "200"; do
        for alt in "12m" "48m"; do
            for dataset in "clustered_1" "clustered_2" "clustered_3" "clustered_4"; do
                if [ -d "planner_evaluation_results/number_of_objects/baseline_${alt}_${number}_${dataset}" ]; then
                    echo "Skipping 'baseline_${alt}_${number}_${dataset}' because it already exists!"
                    continue
                fi

                adaptive_planner orthomosaic_sim \
                    fields/${dataset}.yaml \
                    experiments/number_of_objects/baseline_${alt}_planner_parameters.yaml \
                    --output_folder planner_evaluation_results/number_of_objects/baseline_${alt}_${number}_${dataset} \
                    --executor_config_file experiments/number_of_objects/executor_${number}_objects_${dataset}.yaml
            done
        done
    done
fi

if contains_option "number_of_objects" "$@"; then
    for number in "0" "20" "40" "60" "80" "100" "120" "140" "160" "180" "200"; do
        for dataset in "uniform_1" "uniform_2" "uniform_3" "uniform_4"; do
            if [ -d "planner_evaluation_results/number_of_objects/36m_${number}_${dataset}" ]; then
                echo "Skipping '36m_${number}_${dataset}' because it already exists!"
                continue
            fi

            adaptive_planner orthomosaic_sim \
                fields/${dataset}.yaml \
                experiments/number_of_objects/planner_parameters_uniform.yaml \
                --output_folder planner_evaluation_results/number_of_objects/36m_${number}_${dataset} \
                --base_flight_folder planner_evaluation_results/number_of_objects/baseline_36m_${number}_${dataset} \
                --executor_config_file experiments/number_of_objects/executor_${number}_objects_${dataset}.yaml
        done

        for dataset in "clustered_1" "clustered_2" "clustered_3" "clustered_4"; do
            if [ -d "planner_evaluation_results/number_of_objects/36m_${number}_${dataset}" ]; then
                echo "Skipping '36m_${number}_${dataset}' because it already exists!"
                continue
            fi

            adaptive_planner orthomosaic_sim \
                fields/${dataset}.yaml \
                experiments/number_of_objects/planner_parameters_clustered.yaml \
                --output_folder planner_evaluation_results/number_of_objects/48m_${number}_${dataset} \
                --base_flight_folder planner_evaluation_results/number_of_objects/baseline_48m_${number}_${dataset} \
                --executor_config_file experiments/number_of_objects/executor_${number}_objects_${dataset}.yaml
        done
    done
fi
