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

SCRIPT="${ACCELERATE_DIR}/scripts/train_res_tokenizer.py"

####################
### Set run name ###
####################
RUN_NAME="vq-test"
####################


## change the batch size according to GPU memory
SCRIPT_ARGS="
    config=${ACCELERATE_DIR}/configs/tokenizer/rqgan_tokenizer_10bit_4lvl.yaml \
    training.per_gpu_batch_size=16 \
    training.gradient_accumulation_steps=1 \
    dataset.params.train_shards_path_or_url=./shards/train/imagenet-train-{0000..0008}.tar \
    dataset.params.eval_shards_path_or_url=./shards/val/imagenet-val-0000.tar \
    experiment.save_every=100 \
    experiment.generate_every=100 \
    experiment.eval_every=100 \
    experiment.run_name=${RUN_NAME} \
    experiment.logger=tensorboard \
    model.vq_model.schedule_type='anneal' \
    model.vq_model.schedule_params.anneal_start=100_000 \
    model.vq_model.schedule_params.anneal_end=500_000 \
    "
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"
echo "Running command: $CMD"
$CMD