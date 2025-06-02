source activateEnvironment.sh
GPUS_PER_NODE=2
######################

######################
#### Set network #####
######################
######################

LAUNCHER="accelerate launch \
    --num_processes $((1 * GPUS_PER_NODE)) \
    --num_machines 1 \
    "
ACCELERATE_DIR="/datapool/data2/home/linhw/zhangxiangcheng/DiffAR/PrecisionAR-LFQ/maskbit"
cd $ACCELERATE_DIR

SCRIPT="${ACCELERATE_DIR}/scripts/train_res_tokenizer.py"

####################
### Set run name ###
####################
RUN_NAME="1lvl_test"
####################


## change the batch size according to GPU memory
SCRIPT_ARGS="
    config=${ACCELERATE_DIR}/configs/tokenizer/rqbit_tokenizer_10bit.yaml \
    training.per_gpu_batch_size=16 \
    training.gradient_accumulation_steps=2 \
    dataset.params.train_shards_path_or_url=./shards/train/imagenet-train-{0000..0008}.tar \
    dataset.params.eval_shards_path_or_url=./shards/imagenet-val-0009.tar \
    experiment.save_every=100 \
    experiment.generate_every=100 \
    experiment.eval_every=400 \
    experiment.run_name=${RUN_NAME} \
    "
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"
$CMD