source activateEnvironment.sh
GPUS_PER_NODE=1
######################

######################
#### Set network #####
######################
######################

LAUNCHER="accelerate launch \
    --num_processes $((1 * GPUS_PER_NODE)) \
    --num_machines 1 \
    "
ACCELERATE_DIR="/n/holylfs06/LABS/sham_lab/Users/ydu/zhangxiangcheng/PrecisionAR-LFQ/maskbit"

cd $ACCELERATE_DIR

SCRIPT="${ACCELERATE_DIR}/scripts/train_res_tokenizer.py"

## change the batch size according to GPU memory
SCRIPT_ARGS="
    config=${ACCELERATE_DIR}/configs/tokenizer/rqbit_tokenizer_10bit.yaml \
    training.per_gpu_batch_size=8 \ 
    training.max_train_steps=1_350_000 \
    training.mixed_precision='no' \
    dataset.params.train_shards_path_or_url=${ACCELERATE_DIR}/shards/train/imagenet-train-{0000..0252}.tar  \
    dataset.params.val_shards_path_or_url=${ACCELERATE_DIR}/shards/val/imagenet-val-{0000..0009}.tar \
    experiment.save_every=100 \
    experiment.generate_every=100 \
    experiment.eval_every=200   
    "
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"
$CMD