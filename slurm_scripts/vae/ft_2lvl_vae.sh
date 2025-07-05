#!/bin/bash

#SBATCH --job-name=vae-2-ft
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
#SBATCH --account=kempner_sham_lab

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
RUN_NAME="ft-2lvl-small_lr"
####################


###################
### Config file ###
###################
config_file=$ACCELERATE_DIR/configs/tokenizer/ft_maskbit_tokenizer_10bit_2lvl.yaml
###################


###################
## Model args #####
###################
MODEL_ARGS="model.vq_model.schedule_type=uniform \
    losses.discriminator_start=2000 \
    optimizer.params.learning_rate=5e-5 \
    optimizer.params.discriminator_learning_rate=2e-5 \
    experiment.init_checkpoint=/n/holylfs06/LABS/sham_lab/Users/ydu/zhangxiangcheng/PresionAR-LFQ/maskbit/runs/outputs/rqbit_tokenizer_10bit/ft-2lvl-from_scratch/archive/checkpoint-100000 \
    "
###################

srun bash -c "
    accelerate launch \
    --multi_gpu \
    --num_processes $NUM_PROCESSES \
    $ACCELERATE_DIR/scripts/ft_res_tokenizer.py \
    config=$config_file \
    training.per_gpu_batch_size=32 \
    training.gradient_accumulation_steps=2 \
    experiment.save_every=1_000 \
    experiment.resume=true \
    experiment.run_name=${RUN_NAME} \
    experiment.vqgan_checkpoint=/n/holylfs06/LABS/sham_lab/Users/ydu/zhangxiangcheng/PresionAR-LFQ/ckpts/maskbit_tokenizer_10bit.bin \
    ${MODEL_ARGS} \
    "



