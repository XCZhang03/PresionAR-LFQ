#!/bin/bash

#SBATCH --job-name=adaln-adaln-xxslr
#SBATCH -p kempner_requeue
#SBATCH --mem=256G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=504985967@qq.com
#SBATCH -o status/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e status/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --open-mode=append
#SBATCH --nodes=2                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --cpus-per-task=16           # number of CPU cores per task
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4                # number of GPUs per node
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
# RUN_NAME="adaln-adaln-2lvl-ft"
# RUN_NAME="control-both-2lvl-ft"
# RUN_NAME="adaln-adaln-xslr-lbs-lb1-sb2"
# RUN_NAME="adaln-adaln-resume-xslr-lbs-lb1-sb2"
# RUN_NAME="adaln-adaln-lbs"
RUN_NAME="adaln-adaln-xxslr"
####################

###################
### Config file ###
###################
config_file=$ACCELERATE_DIR/configs/ar/ar_generator_ft_10bit_2lvl.yaml
###################


####################
## Tokenizer ckpt ##
####################
vqgan_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/rqbit_tokenizer_10bit/ft-2lvl-small_lr/archive/checkpoint-200000/ema_model
####################

######################
## Stage model ckpt ##
######################
stage_0_model_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/ckpts/maskbit_generator_10bit-new.bin
######################


#####################
## Model args #######
#####################
# MODEL_ARGS="model.cond_model.context_conditioning=adaln \
#     model.cond_model.label_conditioning=adaln \
#     "
# MODEL_ARGS="model.cond_model.context_conditioning=control \
#     model.cond_model.label_conditioning=both \
#     "
# MODEL_ARGS=""
# MODEL_ARGS="model.cond_model.context_conditioning=adaln \
#     model.cond_model.label_conditioning=adaln \
#     optimizer.params.beta1=0.92 \
#     optimizer.params.beta2=0.95 \
#     training.max_train_steps=200_000 \
#     "
# MODEL_ARGS="model.cond_model.context_conditioning=adaln \
#     model.cond_model.label_conditioning=adaln \
#     optimizer.params.beta1=0.92 \
#     optimizer.params.beta2=0.95 \
#     optimizer.params.learning_rate=6e-5 \
#     training.max_train_steps=200_000 \
#     experiment.init_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/resar_generator_10bit/adaln-adaln-2lvl-ft/archive/checkpoint-100000 \
#     experiment.dont_resume_optimizer=true \
#     experiment.resume_lr_scheduler=false \
#     "
MODEL_ARGS="model.cond_model.context_conditioning=adaln \
    model.cond_model.label_conditioning=adaln \
    optimizer.params.beta1=0.92 \
    optimizer.params.beta2=0.95 \
    optimizer.params.learning_rate=2e-5 \
    training.max_train_steps=400_000 \
    experiment.init_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/resar_generator_10bit/adaln-adaln-resume-xslr-lbs-lb1-sb2/archive/checkpoint-200000 \
    experiment.dont_resume_optimizer=true \
    experiment.resume_lr_scheduler=false \
    "
####################

srun bash -c "
    accelerate launch \
    --multi_gpu \
    --rdzv_backend c10d \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    --machine_rank $SLURM_PROCID \
    $ACCELERATE_DIR/scripts/train_ar.py \
    config=$config_file \
    training.per_gpu_batch_size=32 \
    training.gradient_accumulation_steps=8 \
    experiment.run_name=${RUN_NAME} \
    experiment.vqgan_checkpoint=${vqgan_checkpoint} \
    experiment.eval_every=10_000 \
    model.ar_model.stage_0_model_checkpoint=${stage_0_model_checkpoint} \
    ${MODEL_ARGS} \
    "


