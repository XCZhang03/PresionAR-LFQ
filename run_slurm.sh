#!/bin/bash

#SBATCH --job-name=vae-1
#SBATCH -p gpu
#SBATCH --mem=50G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=504985967@qq.com
#SBATCH -o status/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e status/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --nodes=2                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:2                # number of GPUs per node
#SBATCH -t 0-02:00                  # maximum execution time (HH:MM:SS)
#SBATCH --contiguous

######################
### Set enviroment ###
######################
source activateEnvironment.sh
GPUS_PER_NODE=2
export LOG_LEVEL=INFO
######################

######################
#### Set network #####
######################
# head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
head_node_hostname=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
head_node_ip=$(getent hosts $head_node_hostname | awk '{ print $1 }')
echo "head_node_ip: $head_node_ip"
echo "head_node_hostname: $head_node_hostname"
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500
# Use Infiniband interface for distributed backend
export GLOO_SOCKET_IFNAME=ib0
export NCCL_SOCKET_IFNAME=ib0
######################
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_PROCID: $SLURM_PROCID"
NNODES=$SLURM_NNODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)
echo "NUM_PROCESSES: $NUM_PROCESSES"

ACCELERATE_DIR="/n/holylfs06/LABS/sham_lab/Users/ydu/zhangxiangcheng/PresionAR-LFQ/maskbit"
cd $ACCELERATE_DIR

srun bash -c "accelerate launch \
    --multi_gpu \
    --rdzv_backend c10d \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_PROCID \
    $ACCELERATE_DIR/scripts/train_res_tokenizer.py \
    config=$ACCELERATE_DIR/configs/tokenizer/rqbit_tokenizer_10bit.yaml \
    training.per_gpu_batch_size=32"



