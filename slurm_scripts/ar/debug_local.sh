# !/bin/bash

source activateEnvironment.sh

######################
### Set GPUs #########
######################
GPUS_PER_NODE=1
# export CUDA_VISIBLE_DEVICES=0
######################


LAUNCHER="accelerate launch \
    --num_processes $((1 * GPUS_PER_NODE)) \
    --num_machines 1 \
    "

SCRIPT="${ACCELERATE_DIR}/scripts/train_ar.py"

####################
### Set run name ###
####################
RUN_NAME="test-stage0"
####################

###################
### Config file ###
###################
config_file=$ACCELERATE_DIR/configs/ar/ar_generator_10bit_2lvl.yaml
###################

####################
## Tokenizer ckpt ##
####################
vqgan_checkpoint="/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/rqbit_tokenizer_10bit/ft-2lvl-xs_lr/checkpoints/checkpoint_69/ema_model"
####################


######################
## Stage model ckpt ##
######################
stage_0_model_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/ckpts/maskbit_generator_10bit-new.bin
######################

## change the batch size according to GPU memory
SCRIPT_ARGS="
    config=${config_file} \
    training.per_gpu_batch_size=2 \
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
    "
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"
echo "Running command: $CMD"
$CMD