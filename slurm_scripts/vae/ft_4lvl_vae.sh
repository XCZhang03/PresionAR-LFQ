#!/bin/bash

#SBATCH --job-name=ft-12b-4lvl
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
#SBATCH -t 2-00:00                  # maximum execution time (HH:MM:SS)
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
# RUN_NAME="ft-4lvl-12bit"
# RUN_NAME="ft-4lvl-spectralnorm"
RUN_NAME="ft-4lvl-slr"
####################


###################
### Config file ###
###################
config_file=$ACCELERATE_DIR/configs/tokenizer/rqbit_tokenizer_12bit_4lvl.yaml
###################


###################
## Model args #####
###################
# MODEL_ARGS="losses.discriminator_start=2000 \
#     optimizer.params.learning_rate=5e-5 \
#     optimizer.params.discriminator_learning_rate=5e-5 \
#     training.max_train_steps=200_000 \
#     model.vq_model.entropy_loss_weight=[0.1,0,0,0] \
#     model.discriminator.spectral_norm=True \
#     experiment.vqgan_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/rqbit_tokenizer_12bit/ft-4lvl-spectralnorm/archive/checkpoint-80000/ema_model/pytorch_model.bin \
#     "
MODEL_ARGS="losses.discriminator_start=2000 \
    optimizer.params.learning_rate=5e-5 \
    optimizer.params.discriminator_learning_rate=5e-6 \
    training.max_train_steps=200_000 \
    model.vq_model.entropy_loss_weight=[0.1,0,0,0] \
    model.discriminator.spectral_norm=True \
    losses.lecam_regularization_weight=0.01 \
    experiment.init_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/rqbit_tokenizer_12bit/ft-4lvl-spectralnorm/archive/checkpoint-60000 \
    experiment.dont_resume_optimizer=True \
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
    ${MODEL_ARGS} \
    "



