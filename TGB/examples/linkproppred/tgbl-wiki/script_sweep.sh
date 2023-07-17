#!/bin/bash

# Specify the path to your conda environment
SOURCE_ENV_PATH="/lfs/local/0/zhiyinl//supply-chains/TGB/tgb_env/bin/activate"

# Specify the path to the directory containing your script
SCRIPT_DIR="/lfs/local/0/zhiyinl/supply-chains/TGB/examples/linkproppred/tgbl-wiki"

# Specify weighted & biases parameters
WANDB_KEY="3bc017330056159c6bba091c7e3d5431a8aa45ec"
WANDB_PROJECT="curis-2023-tgb"
WANDB_USERNAME="zhiyinl"

# Loop through the GPU indices and launch the jobs in separate tmux sessions
cd $SCRIPT_DIR
source $SOURCE_ENV_PATH
export PATH="/lfs/local/0/zhiyinl/anaconda3/bin:$PATH"
export WANDB_API_KEY=$WANDB_KEY
CUDA_VISIBLE_DEVICES=$2 wandb agent --project ${WANDB_PROJECT} --entity ${WANDB_USERNAME} $1
