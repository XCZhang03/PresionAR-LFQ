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
ACCELERATE_DIR="/n/holylfs06/LABS/sham_lab/Users/ydu/zhangxiangcheng/PresionAR-LFQ/maskbit"
cd $ACCELERATE_DIR

SCRIPT="${ACCELERATE_DIR}/scripts/train_res_tokenizer.py"

## change the batch size according to GPU memory
SCRIPT_ARGS="
    config=${ACCELERATE_DIR}/configs/tokenizer/rqbit_tokenizer_10bit.yaml \
    training.per_gpu_batch_size=8 \ 
    experiment.save_every=100 \
    experiment.generate_every=100 \
    experiment.eval_every=200   
    "
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"
$CMD