#!/bin/bash

#SBATCH --job-name=eval-ar
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
#SBATCH -t 0-20:00                  # maximum execution time (HH:MM:SS)
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
vqgan_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/rqbit_tokenizer_10bit/4lvl-ft_dec-2m/archive/checkpoint-1800000/ema_model/pytorch_model.bin
# ar_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/resar_generator_10bit/adaln-adaln-resume/checkpoints/checkpoint_50/ema_model/composed_model_0/pytorch_model.bin
# ar_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/resar_generator_10bit/2lvl-bit-group2-both/archive/checkpoint-200000/ema_model/composed_model_0/pytorch_model.bin
# ar_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/resar_generator_10bit/adaln-adaln-resume/checkpoints/checkpoint_50/ema_model/composed_model_2/pytorch_model.bin
ar_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/resar_generator_10bit/4lvl-stage3/checkpoints/checkpoint_23/ema_model/pytorch_model.bin

###################
### Config file ###
###################
# config_file=$ACCELERATE_DIR/configs/ar/ar_generator_10bit_2lvl.yaml
# config_file=$ACCELERATE_DIR/configs/ar/ar_bit_generator_10bit_2lvl.yaml
config_file=$ACCELERATE_DIR/configs/ar/ar_generator_10bit_4lvl.yaml
###################

####################
### Set run name ###
####################
RUN_NAME="eval_ar-${date_str}"
####################


###################
## Model args #####
###################
MODEL_ARGS="model.cond_model.guidance_scale=[2.0,1.0,0] \
    model.cond_model.num_steps=[64,32,2] \
    model.cond_model.randomize_temperature=10 \
    model.base_model.randomize_temperature=10 \
    model.base_model.guidance_scale=4.0 \
    model.base_model.num_steps=64 \
    "
# MODEL_ARGS="model.ar_model.cur_stage=0 \
#     model.base_model.guidance_scale=5.0 \
#     "
###################



srun bash -c "
    python \
    $ACCELERATE_DIR/scripts/eval_ar.py \
    config=$config_file \
    experiment.vqgan_checkpoint=$vqgan_checkpoint \
    experiment.run_name=$RUN_NAME \
    training.per_gpu_batch_size=100 \
    experiment.ar_checkpoint=$ar_checkpoint \
    ${MODEL_ARGS} \
    "



