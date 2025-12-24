#!/bin/bash

#SBATCH --job-name=ft-4lvl
#SBATCH -p kempner_requeue
#SBATCH --mem=256G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=504985967@qq.com
#SBATCH -o status/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e status/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --open-mode=append
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --cpus-per-task=16           # number of CPU cores per task
#SBATCH --gres=gpu:nvidia_h200:4                # number of GPUs per node
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
SITE_DOMAIN="rc.fas.harvard.edu"
MASTER_SHORT="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
if [[ "$MASTER_SHORT" == *".${SITE_DOMAIN}" ]]; then
  export MASTER_ADDR="$MASTER_SHORT"
else
  export MASTER_ADDR="$MASTER_SHORT.${SITE_DOMAIN}"
fi
export MASTER_PORT=29500
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
RUN_NAME="ft-4lvl"
####################

###################
### Config file ###
###################
config_file=$ACCELERATE_DIR/configs/base_gen/maskbit_generator_10bit_4lvl.yaml
###################


####################
## Tokenizer ckpt ##
####################
vqgan_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit/runs/outputs/rqbit_tokenizer_10bit/4level-from_scratch-long/checkpoints/checkpoint_1314/ema_model/pytorch_model.bin
####################


#####################
## Model args #######
#####################
MODEL_ARGS="experiment.mlm_checkpoint=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/ckpts/maskbit_generator_10bit-new.bin \
    training.max_train_steps=450_000 \
    optimizer.params.learning_rate=5e-5 \
    "
####################

srun bash -c "
    accelerate launch \
    --rdzv_backend c10d \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_PROCID \
    $ACCELERATE_DIR/scripts/train_maskbit.py \
    config=$config_file \
    training.per_gpu_batch_size=128 \
    training.gradient_accumulation_steps=2 \
    experiment.run_name=${RUN_NAME} \
    experiment.vqgan_checkpoint=${vqgan_checkpoint} \
    experiment.eval_every=5_000 \
    ${MODEL_ARGS} \
    "


