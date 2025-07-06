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

SCRIPT="${ACCELERATE_DIR}/scripts/ft_res_tokenizer.py"

####################
### Set run name ###
####################
RUN_NAME="ft-test"
####################


## change the batch size according to GPU memory
SCRIPT_ARGS="
    config=${ACCELERATE_DIR}/configs/tokenizer/ft_maskbit_tokenizer_10bit_2lvl.yaml \
    training.per_gpu_batch_size=16 \
    training.gradient_accumulation_steps=1 \
    dataset.params.train_shards_path_or_url=./shards/train/imagenet-train-{0000..0008}.tar \
    dataset.params.eval_shards_path_or_url=./shards/val/imagenet-val-0000.tar \
    experiment.save_every=100 \
    experiment.generate_every=100 \
    experiment.eval_every=100 \
    experiment.run_name=${RUN_NAME} \
    experiment.logger=tensorboard \
    optimizer.params.learning_rate=5e-5 \
    optimizer.params.discriminator_learning_rate=2e-5 \
    experiment.vqgan_checkpoint=/datapool/data2/home/linhw/zhangxiangcheng/DiffAR/PrecisionAR-LFQ/maskbit/maskbit_tokenizer_10bit.bin \
    "
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"
echo "Running command: $CMD"
$CMD