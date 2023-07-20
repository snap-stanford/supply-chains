#!/bin/bash
# printf "Enter your wandb sweep id:\n"
# read WANDB_SWEEPID
WANDB_SWEEPID="6zvc3hcl"

if [[ $USER == "" ]]
then
  WANDB_USERNAME=""
elif [[ $USER == "zhiyinl" ]]
then
  WANDB_USERNAME="zhiyinl"
fi

# Specify the number of GPUs and the desired GPU indices to run on.
NUM_GPUS=2
GPU_INDICES=(0 1)
NUM_AGENTS_PER_GPU=2

# Specify the path of script that tmux sessions run
SCRIPT_DIR="/lfs/local/0/zhiyinl/supply-chains/TGB/examples/linkproppred/tgbl-wiki/script_sweep.sh"

# Loop through the GPU indices and launch the jobs in separate tmux sessions
for ((i=0; i<NUM_GPUS; i++))
do
  for ((j=0; j<NUM_AGENTS_PER_GPU; j++))
  do
    tmux new-session -d -s "gpu_${i}_${j}"
    tmux send-keys -t "gpu_${i}_${j}" "${SCRIPT_DIR} ${WANDB_SWEEPID} ${GPU_INDICES[i]}" Enter
  done
done
