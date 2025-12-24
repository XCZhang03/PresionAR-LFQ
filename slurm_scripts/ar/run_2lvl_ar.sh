#!/bin/bash

#SBATCH --job-name=bit-concat
#SBATCH -p kempner_requeue
#SBATCH --mem=256G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=504985967@qq.com
#SBATCH -o status/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e status/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --open-mode=append
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --cpus-per-task=16           # number of CPU cores per task
#SBATCH --gres=gpu:nvidia_h200:4                # number of GPUs per node
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
SITE_DOMAIN="rc.fas.harvard.edu"
MASTER_SHORT="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
if [[ "$MASTER_SHORT" == *".${SITE_DOMAIN}" ]]; then
  export MASTER_ADDR="$MASTER_SHORT"
else
  export MASTER_ADDR="$MASTER_SHORT.${SITE_DOMAIN}"
fi
export MASTER_PORT=29500
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
# RUN_NAME="adaln-adaln-resume-llr"
# RUN_NAME="adaln-adaln-resume"
# RUN_NAME="2lvl-bit-mask_token-adaln"
# RUN_NAME="2lvl-bit-group2-both"
# RUN_NAME="adaln-adaln-0929"
RUN_NAME="2lvl-bit-group2-concat"
####################

###################
### Config file ###
###################
# config_file=$ACCELERATE_DIR/configs/ar/ar_generator_10bit_2lvl.yaml
config_file=$ACCELERATE_DIR/configs/ar/ar_bit_generator_10bit_2lvl.yaml
###################


####################
## Tokenizer ckpt ##
####################
vqgan_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/rqbit_tokenizer_10bit/2level-resume/checkpoints/checkpoint_955/ema_model
####################

######################
## Stage model ckpt ##
######################
stage_0_model_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/resar_generator_10bit/2lvl-bit-group2-both/archive/checkpoint-200000/ema_model/base_model.bin
######################


#####################
## Model args #######
#####################
# MODEL_ARGS="model.ar_model.stage_1_model_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/conditional_generator_10bit/2lvl-adaln-adaln-long/checkpoints/checkpoint_219/ema_model \
#     training.max_train_steps=400_000 \
#     optimizer.params.learning_rate=2e-4"
# MODEL_ARGS="model.ar_model.stage_1_model_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/resar_generator_10bit/adaln-adaln-resume/checkpoints/checkpoint_50/ema_model/stage_1_model.bin \
#     training.max_train_steps=400_000 \
#     optimizer.params.learning_rate=5e-5 \
#     model.cond_model.guidance_scale=1.0 \
#     model.cond_model.randomize_temperature=15 \
#     model.base_model.randomize_temperature=15 \
#     model.base_model.guidance_scale=4.0 \
#     "
# MODEL_ARGS="model.base_model.guidance_scale=3.0"
MODEL_ARGS="model.cond_model.mask_token_embedding=true \
    model.cond_model.mask_pos_embedding=false \
    model.cond_model.label_conditioning=concat \
    model.cond_model.codebook_splits=2 \
    model.ar_model.stage_0_model_checkpoint=${stage_0_model_checkpoint} \
    model.ar_model.stage_1_model_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/resar_generator_10bit/2lvl-bit-group2-concat-old/checkpoints/checkpoint_37/ema_model/stage_1_model.bin \
    "
####################

srun bash -c "
    accelerate launch \
    --multi_gpu \
    --rdzv_backend c10d \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_PROCID \
    $ACCELERATE_DIR/scripts/train_ar.py \
    config=$config_file \
    training.per_gpu_batch_size=128 \
    training.gradient_accumulation_steps=4 \
    experiment.run_name=${RUN_NAME} \
    experiment.vqgan_checkpoint=${vqgan_checkpoint} \
    experiment.eval_every=10_000 \
    ${MODEL_ARGS} \
    "


