#!/bin/bash

#SBATCH --job-name=multinode
#SBATCH -D .
#SBATCH --output=O-%x.%j
#SBATCH --error=E-%x.%j
#SBATCH --nodes=4                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:4                # number of GPUs per node
#SBATCH --cpus-per-task=160         # number of cores per tasks
#SBATCH --time=01:59:00             # maximum execution time (HH:MM:SS)

######################
### Set enviroment ###
######################
source activateEnvironment.sh
export GPUS_PER_NODE=4
######################

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################

export LAUNCHER="accelerate launch \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    "
export ACCELERATE_DIR="$/home/linhw/zxc/PresionAR-LFQ/maskbit"
export SCRIPT="${ACCELERATE_DIR}/sscripts/train_res_tokenizer.py"

## change the batch size according to GPU memory
SCRIPT_ARGS="
    config=${ACCELERATE_DIR}/configs/tokenizer/rqbit_tokenizer_10bit.yaml \
    training.per_gpu_batch_size=32 \ 
    training.max_train_steps=300_000 \
    training.mixed_precision='no' \
    dataset.params.train_shards_path_or_url=./shards/train/imagenet-train-{0000..0252}.tar  \
    dataset.params.val_shards_path_or_url=./shards/val/imagenet-val-{0000..0009}.tar \
    experiment.save_every=20_000 \
    experiment.generate_every=2000 \
    experiment.eval_every=20_000 
    "
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"
srun $CMD