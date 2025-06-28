#!/bin/bash

#SBATCH --job-name=vae-4-large_scale
#SBATCH -p kempner_requeue
#SBATCH --mem=100G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=504985967@qq.com
#SBATCH -o status/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e status/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --cpus-per-task=16           # number of CPU cores per task
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4               # number of GPUs per node
#SBATCH -t 6-00:00                  # maximum execution time (HH:MM:SS)
#SBATCH --contiguous
#SBATCH --account=kempner_sham_lab

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
# RUN_NAME="4level-from_scratch-long"
# RUN_NAME="4level-resume_from_1lvl"
# RUN_NAME="4lvl-from_scratch-half_weight"
# RUN_NAME="4lvl-from_scratch-base_3"
# RUN_NAME="4lvl-from_scratch-base_4"
# RUN_NAME="4lvl-from_scratch-half_quantize_weight"
# RUN_NAME="4lvl-half_entropy_gamma-from_scratch"
# RUN_NAME="4lvl-2variant-resume"
RUN_NAME="4lvl-large_scale"
####################


###################
### Config file ###
###################
config_file=$ACCELERATE_DIR/configs/tokenizer/rqbit_tokenizer_10bit_4lvl.yaml
###################


###################
## Model args #####
###################
# MODEL_ARGS="model.vq_model.schedule_type=uniform"
# MODEL_ARGS="model.vq_model.schedule_type=weighted \
#     model.vq_model.weights=[3,1,1,1] \
#     "
# MODEL_ARGS="
#     model.vq_model.entropy_gamma=[1.0,0.5,0.25,0.125] \
#     "
# MODEL_ARGS="
#     model.vq_model.variants=[2,2,2,2] \
#     experiment.init_checkpoint=/n/holylfs06/LABS/sham_lab/Users/ydu/zhangxiangcheng/PresionAR-LFQ/maskbit/runs/outputs/rqbit_tokenizer_10bit/2level-2variant-from_scratch-long/checkpoints/checkpoint_86 \
#     "
MODEL_ARGS="
    model.vq_model.scales=[1,0.8,0.6,0.4] \
    "
###################

srun bash -c "
    accelerate launch \
    --multi_gpu \
    --num_processes $NUM_PROCESSES \
    $ACCELERATE_DIR/scripts/train_res_tokenizer.py \
    config=$config_file \
    training.per_gpu_batch_size=32 \
    training.gradient_accumulation_steps=2 \
    experiment.save_every=1_000 \
    experiment.resume=true \
    experiment.run_name=${RUN_NAME} \
    ${MODEL_ARGS} \
    "



