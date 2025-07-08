# !/bin/bash

source activateEnvironment.sh

export ACCELERATE_DIR="/datapool/data2/home/linhw/zhangxiangcheng/DiffAR/PrecisionAR-LFQ/maskbit"
cd $ACCELERATE_DIR
######################
### Set GPUs #########
######################
GPUS_PER_NODE=1
export CUDA_VISIBLE_DEVICES=6,7
######################


LAUNCHER="accelerate launch \
    --num_processes $((1 * GPUS_PER_NODE)) \
    --num_machines 1 \
    "

SCRIPT="${ACCELERATE_DIR}/scripts/train_ar.py"

####################
### Set run name ###
####################
RUN_NAME="test-stage1-bit"
####################

###################
### Config file ###
###################
config_file=$ACCELERATE_DIR/configs/ar/ar_generator_10bit_4lvl.yaml
###################

####################
## Tokenizer ckpt ##
####################
vqgan_checkpoint="/datapool/data2/home/linhw/zhangxiangcheng/DiffAR/PrecisionAR-LFQ/maskbit/runs/outputs/rqbit_tokenizer_10bit/4lvl-test-loss_weight/archive/checkpoint-200/ema_model"
####################


######################
## Stage model ckpt ##
######################
stage_0_model_checkpoint=runs/outputs/maskbit_generator_10bit/test-base-4lvl-2variant/checkpoints/checkpoint_0/ema_model
######################

## change the batch size according to GPU memory
SCRIPT_ARGS="
    config=${config_file} \
    training.per_gpu_batch_size=32 \
    training.gradient_accumulation_steps=1 \
    dataset.params.train_shards_path_or_url=./shards/train/imagenet-train-{0000..0008}.tar \
    dataset.params.eval_shards_path_or_url=./shards/val/imagenet-val-0000.tar \
    experiment.save_every=100 \
    experiment.generate_every=100 \
    experiment.eval_every=100 \
    experiment.resume=true \
    experiment.run_name=${RUN_NAME} \
    experiment.logger=tensorboard \
    experiment.vqgan_checkpoint=${vqgan_checkpoint} \
    model.cond_model.num_steps=2 \
    model.base_model.num_steps=2 \
    model.ar_model.stage_0_model_checkpoint=${stage_0_model_checkpoint} \
    model.cond_model.model_cls=cond_lfq_bert \
    "
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"
echo "Running command: $CMD"
$CMD