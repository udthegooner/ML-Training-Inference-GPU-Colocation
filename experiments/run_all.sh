#!/bin/bash

# --- Configurations ---
# Ensure the script knows where the project root is
PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Output directories
LOG_DIR="experiments/logs"
mkdir -p $LOG_DIR
RESULTS_DIR="experiments/results"
mkdir -p $RESULTS_DIR

# --- MPS Management ---
# Enable MPS to allow multiple processes to share the GPU
# TODO: check
export CUDA_VISIBLE_DEVICES=0 # Use only GPU 0
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS # Set GPU to exclusive process mode
nvidia-cuda-mps-control -d # Start MPS control daemon

# Cleanup function to stop MPS when the script exits or is killed
cleanup() {
    echo "Stopping MPS control daemon..."
    echo quit | nvidia-cuda-mps-control
    nvidia-smi -i 0 -c DEFAULT # Reset GPU to default mode
}
trap cleanup EXIT

# --- Experiment Execution ---
models=('cyclegan' 'lstm' 'resnet50' 's2s' 'dqn')
outer_types=('training' 'training' 'training' 'training' 'training')
inner_types=('inference' 'inference' 'inference' 'inference' 'inference')
batch_sizes=(1 20 32 32 32)
num_steps=(50 50 50 50 50)

echo "Starting GPU Colocation Analysis..."
echo "Using CUDA MPS: $(nvidia-smi -q | grep 'MPS Mode')" # TODO: check

# --- Execution Loop ---
for (( i=0; i<${#models[@]}; i++ )); do
    for (( j=0; j<${#models[@]}; j++ )); do

        MODEL1=${models[i]}
        MODEL2=${models[j]}

        # Define log paths
        LOG1="$LOG_DIR/outer_${MODEL1}_${outer_types[i]}.log"
        LOG2="$LOG_DIR/inner_${MODEL2}_${inner_types[j]}.log"
        OUTPUT_JSON="$RESULTS_DIR/${MODEL1}_${outer_types[i]}_${MODEL2}_${inner_types[j]}.json"

        echo "------------------------------------------------"
        echo "Pair: ${MODEL1} (${outer_types[i]}) + ${MODEL2} (${inner_types[j]})"
        echo "------------------------------------------------"

        # Run main.py
        # TODO: check args
        python3 core/main.py \
            -- MPS 1 \
            -c 1 \
            -m "$MODEL1,$MODEL2" \
            -b "${batch_sizes[i]},${batch_sizes[j]}" \
            -n "${num_steps[i]},${num_steps[j]}" \
            -t "${outer_types[i]},${inner_types[j]}" \
            --log_path1 "$LOG1" \
            --log_path2 "$LOG2" \
            --output_json "$OUTPUT_JSON"
        
        # Parse logs to extract metrics
        python3 core/parse_logs.py \
            --log_path1 "$LOG1" \
            --log_path2 "$LOG2" \
            --output_json "$OUTPUT_JSON"
    done
done

echo "************************************************"
echo "GPU Colocation Analysis Completed."
echo "Results are saved in the '$RESULTS_DIR' directory."
echo "************************************************"