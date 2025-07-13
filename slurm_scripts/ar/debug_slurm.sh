#!/bin/bash

#SBATCH --job-name=ar-debug
#SBATCH -p kempner_requeue
#SBATCH --mem=100G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=504985967@qq.com
#SBATCH -o status/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e status/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --cpus-per-task=8           # number of CPU cores per task
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:2                # number of GPUs per node
#SBATCH -t 0-02:00                  # maximum execution time (HH:MM:SS)
#SBATCH --contiguous
#SBATCH --account=kempner_ydu_lab


######################
### Set enviroment ###
######################
source activateEnvironment.sh
GPUS_PER_NODE=2
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
RUN_NAME="test_ar_eval"
####################

###################
### Config file ###
###################
config_file=$ACCELERATE_DIR/configs/ar/ar_generator_10bit_2lvl.yaml
###################


####################
## Tokenizer ckpt ##
####################
vqgan_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/rqbit_tokenizer_10bit/ft-2lvl-small_lr/archive/checkpoint-200000/ema_model
####################

######################
## Stage model ckpt ##
######################
stage_0_model_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/ckpts/maskbit_generator_10bit-new.bin
######################

srun bash -c "
    accelerate launch \
    --rdzv_backend c10d \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    --machine_rank $SLURM_PROCID \
    $ACCELERATE_DIR/scripts/train_ar.py \
    config=$config_file \
    training.per_gpu_batch_size=4 \
    training.gradient_accumulation_steps=1 \
    experiment.save_every=200 \
    experiment.generate_every=100 \
    experiment.eval_gen_every=100 \
    experiment.eval_loss_every=200 \
    experiment.resume=true \
    experiment.run_name=${RUN_NAME} \
    experiment.vqgan_checkpoint=${vqgan_checkpoint} \
    model.ar_model.stage_0_model_checkpoint=${stage_0_model_checkpoint} \
    "


