#!/bin/bash

#SBATCH --job-name=vae-2-half_ent
#SBATCH -p kempner_requeue
#SBATCH --mem=100G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=504985967@qq.com
#SBATCH -o status/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e status/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --open-mode=append
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --cpus-per-task=16           # number of CPU cores per task
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4                # number of GPUs per node
#SBATCH -t 4-00:00                  # maximum execution time (HH:MM:SS)
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
# RUN_NAME="2level-mixed_after_1lvl-long"
# RUN_NAME="2level-mixed_from_scratch-long"
# RUN_NAME="2level-2variant-from_scratch-long"
# RUN_NAME="2level-large_batch"
# RUN_NAME="2level-ft-decoder-0"
# RUN_NAME="2level-ft-decoder-1"
# RUN_NAME="2level-ft-decoder-0-resume_ema"
# RUN_NAME="2level-ft-decoder-0-restart_ema"
# RUN_NAME="2level-resume"
# RUN_NAME="2lvl-resume_1lvl_long"
# RUN_NAME="2lvl-half_ent"
RUN_NAME="2lvl-half_ent_rev"
####################


###################
### Config file ###
###################
config_file=$ACCELERATE_DIR/configs/tokenizer/rqbit_tokenizer_10bit_2lvl.yaml
###################


###################
## Model args #####
###################
# MODEL_ARGS="model.vq_model.schedule_type=uniform"
# MODEL_ARGS="training.max_train_steps=2_500_000 \
#     experiment.init_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/rqbit_tokenizer_10bit/2level-mixed_from_scratch-long/checkpoints/checkpoint_597 \
#     "
# MODEL_ARGS="model.vq_model.finetune_decoder=true \
#     model.vq_model.schedule_type="weighted"  \
#     model.vq_model.weights=[1,0] \
#     model.vq_model.restart_ema=true \
#     experiment.init_checkpoint=/n/holylfs06/LABS/sham_lab/Users/ydu/zhangxiangcheng/PresionAR-LFQ/maskbit/runs/outputs/rqbit_tokenizer_10bit/2level-mixed_from_scratch-long/archive/checkpoint-700000 \
#     "
# MODEL_ARGS="training.max_train_steps=2_500_000 \
#     experiment.init_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/rqbit_tokenizer_10bit/1level-long/archive/checkpoint_502 \
#     "
# MODEL_ARGS="training.max_train_steps=2_500_000 \
#     model.vq_model.entropy_gamma=[1.0,0.0] \
#     experiment.init_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/rqbit_tokenizer_10bit/1level-long/archive/checkpoint_502 \
#     "
MODEL_ARGS="training.max_train_steps=2_500_000 \
    model.vq_model.entropy_gamma=[0.0,1.0] \
    experiment.init_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/rqbit_tokenizer_10bit/1level-long/archive/checkpoint_502 \
    "
###################




srun bash -c "
    accelerate launch \
    --multi_gpu \
    --rdzv_backend c10d \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    --machine_rank $SLURM_PROCID \
    $ACCELERATE_DIR/scripts/train_res_tokenizer.py \
    config=$config_file \
    training.per_gpu_batch_size=32 \
    training.gradient_accumulation_steps=4 \
    experiment.save_every=500 \
    experiment.resume=true \
    experiment.run_name=${RUN_NAME} \
    ${MODEL_ARGS} \
    "



