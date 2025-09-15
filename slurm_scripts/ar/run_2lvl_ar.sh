#!/bin/bash

#SBATCH --job-name=adaln-adaln
#SBATCH -p kempner_requeue
#SBATCH --mem=256G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=504985967@qq.com
#SBATCH -o status/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e status/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --open-mode=append
#SBATCH --nodes=2                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --cpus-per-task=16           # number of CPU cores per task
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4                # number of GPUs per node
#SBATCH -t 2-00:00                  # maximum execution time (HH:MM:SS)
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
RUN_NAME="adaln-adaln-resume-llr"
####################

###################
### Config file ###
###################
config_file=$ACCELERATE_DIR/configs/ar/ar_generator_10bit_2lvl.yaml
###################


####################
## Tokenizer ckpt ##
####################
vqgan_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/rqbit_tokenizer_10bit/2level-resume/checkpoints/checkpoint_955/ema_model
####################

######################
## Stage model ckpt ##
######################
stage_0_model_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/maskbit_generator_10bit/test-2lvl-base/archive/checkpoint-1200000/ema_model/pytorch_model.bin
######################


#####################
## Model args #######
#####################
MODEL_ARGS="model.ar_model.stage_1_model_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/conditional_generator_10bit/2lvl-adaln-adaln-long/checkpoints/checkpoint_219/ema_model \
    training.max_train_steps=600_000 \
    optimizer.params.learning_rate=2e-4"
####################

srun bash -c "
    accelerate launch \
    --multi_gpu \
    --rdzv_backend c10d \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    --machine_rank $SLURM_PROCID \
    $ACCELERATE_DIR/scripts/train_ar.py \
    config=$config_file \
    training.per_gpu_batch_size=64 \
    training.gradient_accumulation_steps=4 \
    experiment.run_name=${RUN_NAME} \
    experiment.vqgan_checkpoint=${vqgan_checkpoint} \
    experiment.eval_every=10_000 \
    model.ar_model.stage_0_model_checkpoint=${stage_0_model_checkpoint} \
    ${MODEL_ARGS} \
    "


