#!/bin/bash

#SBATCH --job-name=vae-2-ft-resume_disc
#SBATCH -p kempner_requeue
#SBATCH --mem=100G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=504985967@qq.com
#SBATCH -o status/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e status/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --open-mode=append
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
# RUN_NAME="ft-2lvl-small_lr"
# RUN_NAME="ft-2lvl-from_scratch"
RUN_NAME="ft-2lvl-resume_disc"
####################


###################
### Config file ###
###################
config_file=$ACCELERATE_DIR/configs/tokenizer/ft_maskbit_tokenizer_12bit_2lvl.yaml
###################


###################
## Model args #####
###################
# MODEL_ARGS="model.vq_model.schedule_type=uniform"
# MODEL_ARGS="model.vq_model.schedule_type=uniform \
#     losses.discriminator_start=2000 \
#     optimizer.params.learning_rate=5e-5 \
#     optimizer.params.discriminator_learning_rate=2e-5 \
#     experiment.init_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/rqbit_tokenizer_10bit/ft-2lvl-from_scratch/archive/checkpoint-100000 \
#     experiment.dont_resume_optimizer=true \
#     experiment.resume_lr_scheduler=false \
#     "
# MODEL_ARGS="model.vq_model.schedule_type=uniform \
#     losses.discriminator_start=2000 \
#     optimizer.params.learning_rate=2e-5 \
#     optimizer.params.discriminator_learning_rate=1e-5 \
#     experiment.init_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/rqbit_tokenizer_10bit/ft-2lvl-small_lr/archive/checkpoint-200000 \
#     experiment.dont_resume_optimizer=true \
#     experiment.resume_lr_scheduler=false \
#     "
# MODEL_ARGS="model.vq_model.schedule_type=uniform \
#     losses.discriminator_start=2000 \
#     optimizer.params.learning_rate=4e-5 \
#     optimizer.params.discriminator_learning_rate=1e-5 \
#     training.max_train_steps=300_000 \
#     "
MODEL_ARGS="model.vq_model.schedule_type=uniform \
    losses.discriminator_start=2000 \
    optimizer.params.learning_rate=5e-5 \
    optimizer.params.discriminator_learning_rate=2e-5 \
    training.max_train_steps=300_000 \
    experiment.loss_module=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/rqbit_tokenizer_10bit/2level-resume/archive/checkpoint-1100000/model_1.safetensors \
    "
###################

srun bash -c "
    accelerate launch \
    --multi_gpu \
    --num_processes $NUM_PROCESSES \
    $ACCELERATE_DIR/scripts/ft_res_tokenizer.py \
    config=$config_file \
    training.per_gpu_batch_size=16 \
    training.gradient_accumulation_steps=4 \
    experiment.save_every=1_000 \
    experiment.resume=true \
    experiment.run_name=${RUN_NAME} \
    experiment.vqgan_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/rqbit_tokenizer_12bit/ft-2lvl-xs_lr/archive/checkpoint-60000/ema_model/pytorch_model.bin \
    ${MODEL_ARGS} \
    "



