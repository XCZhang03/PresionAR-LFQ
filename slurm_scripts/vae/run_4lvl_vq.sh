#!/bin/bash

#SBATCH --job-name=vq-4-512
#SBATCH -p kempner_requeue
#SBATCH --mem=100G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=504985967@qq.com
#SBATCH -o status/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e status/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --cpus-per-task=16           # number of CPU cores per task
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4               # number of GPUs per node
#SBATCH -t 6-00:00                  # maximum execution time (HH:MM:SS)
#SBATCH --contiguous
#SBATCH --account=kempner_ydu_lab

######################
### Set enviroment ###
######################
source activateEnvironment.sh
GPUS_PER_NODE=4
export LOG_LEVEL=INFO
######################

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################

######################
## Set launcher ######
######################
NNODES=$SLURM_NNODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)
######################

####################
### Set run name ###
####################
# RUN_NAME="4lvl-512"
RUN_NAME="4lvl-512-anneal"
####################


###################
### Config file ###
###################
config_file=$ACCELERATE_DIR/configs/tokenizer/rqgan_tokenizer_10bit_4lvl.yaml
###################


###################
## Model args #####
###################
# MODEL_ARGS="model.vq_model.schedule_type=uniform"
MODEL_ARGS="model.vq_model.schedule_type='anneal' \
    model.vq_model.schedule_params.anneal_start=100_000 \
    model.vq_model.schedule_params.anneal_end=500_000 \
    "
###################

srun bash -c "
    accelerate launch \
    --num_processes $NUM_PROCESSES \
    $ACCELERATE_DIR/scripts/train_res_tokenizer.py \
    config=$config_file \
    training.per_gpu_batch_size=16 \
    training.gradient_accumulation_steps=4 \
    experiment.save_every=1_000 \
    experiment.resume=true \
    experiment.run_name=${RUN_NAME} \
    ${MODEL_ARGS} \
    "



