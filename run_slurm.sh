#!/bin/bash

#SBATCH --job-name=vae-1
#SBATCH -p gpu_test
#SBATCH --mem=100G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=504985967@qq.com
#SBATCH -o status/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e status/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --nodes=2                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:1                # number of GPUs per node
#SBATCH -t 2-00:00                  # maximum execution time (HH:MM:SS)
#SBATCH --contiguous

######################
### Set enviroment ###
######################
source activateEnvironment.sh
GPUS_PER_NODE=1
export LOG_LEVEL=INFO
######################

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################
NNODES=$SLURM_NNODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

ACCELERATE_DIR="/n/holylfs06/LABS/sham_lab/Users/ydu/zhangxiangcheng/PresionAR-LFQ/maskbit"
cd $ACCELERATE_DIR

srun bash -c "
    accelerate launch \
    --multi_gpu \
    --rdzv_backend c10d \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    --machine_rank $SLURM_PROCID \
    $ACCELERATE_DIR/scripts/train_res_tokenizer.py \
    config=$ACCELERATE_DIR/configs/tokenizer/rqbit_tokenizer_10bit.yaml \
    training.per_gpu_batch_size=8 \
    experiment.save_every=100 \
    "



