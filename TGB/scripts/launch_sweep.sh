#!/bin/bash
read -p "Enter your wandb sweep id: " WANDB_SWEEPID

# Specify the number of GPUs and the desired GPU indices to run on.
printf "Enter your desired GPU indices to run on (split with a whitespace)\n"
read -a GPU_INDICES
NUM_GPUS=${#GPU_INDICES[@]}
printf "You are using ${NUM_GPUS} GPU(s)\n"
read -p "Enter your desired number of agents per GPU: " NUM_AGENTS_PER_GPU
read -p "Enter your file's directory: " FILE_DIR # Directory to tgn.py, for example

# Specify the path of script that tmux sessions run
SCRIPT_DIR="/lfs/local/0/zhiyinl/supply-chains/TGB/scripts/sweep.sh"

# Loop through the GPU indices and launch the jobs in separate tmux sessions
for ((i=0; i<NUM_GPUS; i++))
do
  for ((j=0; j<NUM_AGENTS_PER_GPU; j++))
  do
    tmux new-session -d -s "gpu_${GPU_INDICES[i]}_${j}"
    tmux send-keys -t "gpu_${GPU_INDICES[i]}_${j}" "${SCRIPT_DIR} ${WANDB_SWEEPID} ${GPU_INDICES[i]} ${FILE_DIR}" Enter
  done
done
