#!/bin/bash

#SBATCH --job-name=cond-gen-debug
#SBATCH -p kempner_h100
#SBATCH --mem=100G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=504985967@qq.com
#SBATCH -o status/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e status/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --cpus-per-task=8           # number of CPU cores per task
#SBATCH --gres=gpu:1                # number of GPUs per node
#SBATCH -t 0-01:00                  # maximum execution time (HH:MM:SS)
#SBATCH --contiguous
#SBATCH --account=kempner_sham_lab

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

######################
## Set launcher ######
######################
NNODES=$SLURM_NNODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)
######################

####################
### Set run name ###
####################
RUN_NAME="test_attn"
####################

###################
### Config file ###
###################
config_file=$ACCELERATE_DIR/configs/cond_gen/cond_generator_10bit_2lvl.yaml
###################


####################
## Tokenizer ckpt ##
####################
vqgan_checkpoint=/n/holylfs06/LABS/sham_lab/Users/ydu/zhangxiangcheng/PresionAR-LFQ/maskbit/runs/outputs/rqbit_tokenizer_10bit/2level-mixed_from_scratch-long/archive/checkpoint-800000/ema_model
####################

srun bash -c "
    accelerate launch \
    $ACCELERATE_DIR/scripts/train_cond_mlm.py \
    config=$config_file \
    training.per_gpu_batch_size=64 \
    training.gradient_accumulation_steps=1 \
    experiment.save_every=100 \
    experiment.generate_every=100 \
    experiment.eval_every=200 \
    experiment.resume=true \
    experiment.run_name=${RUN_NAME} \
    experiment.vqgan_checkpoint=${vqgan_checkpoint} \
    training.mixed_precision="bf16" \
    "


