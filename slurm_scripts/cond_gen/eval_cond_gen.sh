#!/bin/bash

#SBATCH --job-name=eval-vae-2lvl
#SBATCH -p kempner_requeue
#SBATCH --mem=100G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=504985967@qq.com
#SBATCH -o status/myeval_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e status/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --nodes=1                  # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --cpus-per-task=16           # number of CPU cores per task
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1                # number of GPUs per node
#SBATCH -t 0-01:00                  # maximum execution time (HH:MM:SS)
#SBATCH --contiguous
#SBATCH --account=kempner_sham_lab

######################
### Set enviroment ###
######################
source activateEnvironment.sh
GPUS_PER_NODE=1
export LOG_LEVEL=INFO
######################


#######################
#### checkpoint #######
#######################
vqgan_checkpoint=/n/holylfs06/LABS/sham_lab/Users/ydu/zhangxiangcheng/PresionAR-LFQ/maskbit/runs/outputs/rqbit_tokenizer_10bit/2level-mixed_from_scratch-long/archive/checkpoint-800000/ema_model
mlm_checkpoint=/n/holylfs06/LABS/sham_lab/Users/ydu/zhangxiangcheng/PresionAR-LFQ/maskbit/runs/outputs/conditional_generator_10bit/2lvl-adaln-adaln/archive/checkpoint-300000/ema_model
#######################

###################
### Config file ###
###################
config_file=$ACCELERATE_DIR/configs/cond_gen/cond_generator_10bit_2lvl.yaml
###################

####################
### Set run name ###
####################
RUN_NAME="2level-eval-cond_gen"
####################


srun bash -c "
    accelerate launch \
    --num_processes 1 \
    $ACCELERATE_DIR/scripts/eval_cond_mlm.py \
    config=$config_file \
    experiment.vqgan_checkpoint=$vqgan_checkpoint \
    experiment.run_name=$RUN_NAME \
    training.per_gpu_batch_size=100 \
    experiment.mlm_checkpoint=$mlm_checkpoint \
    model.mlm_model.num_steps=4 \
    "



