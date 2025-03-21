source activateEnvironment.sh
export GPUS_PER_NODE=4
######################

######################
#### Set network #####
######################
######################

export LAUNCHER="accelerate launch \
    --num_processes $((1 * GPUS_PER_NODE)) \
    --num_machines 1 \
    "
export ACCELERATE_DIR="/home/linhw/zhangxiangcheng/DiffAR/PrecisionAR-LFQ/maskbit"
export SCRIPT="${ACCELERATE_DIR}/scripts/train_res_tokenizer.py"

## change the batch size according to GPU memory
SCRIPT_ARGS="
    config=${ACCELERATE_DIR}/configs/tokenizer/rqbit_tokenizer_10bit.yaml \
    training.per_gpu_batch_size=1 \ 
    training.max_train_steps=300_000 \
    training.mixed_precision='no' \
    dataset.params.train_shards_path_or_url=./shards/train/imagenet-train-{0000..0008}.tar  \
    dataset.params.val_shards_path_or_url=./shards/val/imagenet-val-0000.tar \
    experiment.save_every=100 \
    experiment.generate_every=100 \
    experiment.eval_every=200 
    "
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"
$CMD