#!/bin/bash

# Specify the path to your conda environment
SOURCE_ENV_PATH="/lfs/local/0/zhiyinl/supply-chains/TGB/tgb_env/bin/activate"

# Specify weighted & biases parameters
WANDB_KEY="3bc017330056159c6bba091c7e3d5431a8aa45ec"

# Get variables
WANDB_SWEEPID=$1
GPU_IDX=$2
FILE_DIR=$3

# Loop through the GPU indices and launch the jobs in separate tmux sessions
cd $FILE_DIR
source $SOURCE_ENV_PATH
export PATH="/lfs/local/0/zhiyinl/anaconda3/bin:$PATH"
export WANDB_API_KEY=$WANDB_KEY
CUDA_VISIBLE_DEVICES=$GPU_IDX wandb agent ${WANDB_SWEEPID}