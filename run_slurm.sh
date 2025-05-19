#!/bin/bash

#SBATCH --job-name=vae-1
#SBATCH -p gpu_test
#SBATCH --mem=50G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=504985967@qq.com
#SBATCH -o status/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e status/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --nodes=2                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:1                # number of GPUs per node
#SBATCH -t 0-01:00                  # maximum execution time (HH:MM:SS)

######################
### Set enviroment ###
######################
source activateEnvironment.sh
GPUS_PER_NODE=1
######################

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_PROCID: $SLURM_PROCID"
NNODES=$SLURM_NNODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)
echo "NUM_PROCESSES: $NUM_PROCESSES"

LAUNCHER="accelerate launch \
    --multi_gpu \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    --machine_rank $SLURM_PROCID \

    "
ACCELERATE_DIR="/n/holylfs06/LABS/sham_lab/Users/ydu/zhangxiangcheng/PresionAR-LFQ/maskbit"
cd $ACCELERATE_DIR
SCRIPT="${ACCELERATE_DIR}/scripts/train_res_tokenizer.py"

## change the batch size according to GPU memory
SCRIPT_ARGS="
    config=${ACCELERATE_DIR}/configs/tokenizer/rqbit_tokenizer_10bit.yaml \
    training.per_gpu_batch_size=8 \
    "
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"
srun $CMD