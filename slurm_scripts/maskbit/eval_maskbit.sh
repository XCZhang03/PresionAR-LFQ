#!/bin/bash

#SBATCH --job-name=eval-maskbit
#SBATCH -p kempner_h100
#SBATCH --mem=100G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=504985967@qq.com
#SBATCH -o status/myeval_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e status/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --open-mode=append
#SBATCH --nodes=1                  # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --cpus-per-task=16           # number of CPU cores per task
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1                # number of GPUs per node
#SBATCH -t 0-12:00                  # maximum execution time (HH:MM:SS)
#SBATCH --contiguous
#SBATCH --account=kempner_ydu_lab

######################
### Set enviroment ###
######################
source activateEnvironment.sh
GPUS_PER_NODE=1
export LOG_LEVEL=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
date_str=$(date +%Y-%m-%d_%H-%M-%S)
######################


#######################
#### checkpoint #######
#######################
vqgan_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/rqbit_tokenizer_10bit/2level-resume/checkpoints/checkpoint_955/ema_model
ar_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/maskbit_generator_10bit/test-2lvl-base/archive/checkpoint-1200000/ema_model/pytorch_model.bin

###################
### Config file ###
###################
config_file=$ACCELERATE_DIR/configs/base_gen/maskbit_generator_10bit_2lvl.yaml

####################
### Set run name ###
####################
RUN_NAME="eval_base_ar-${date_str}"
####################





srun bash -c "
    python \
    $ACCELERATE_DIR/scripts/eval_maskbit.py \
    --config=$config_file \
    --tokenizer=$vqgan_checkpoint \
    --generator=$ar_checkpoint \
    --batchsize=100 \
    "



