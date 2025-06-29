#!/bin/bash

#SBATCH --job-name=embed-concat-2lvl
#SBATCH -p kempner_requeue
#SBATCH --mem=100G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=504985967@qq.com
#SBATCH -o status/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e status/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --nodes=4                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --cpus-per-task=8           # number of CPU cores per task
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4                # number of GPUs per node
#SBATCH -t 2-00:00                  # maximum execution time (HH:MM:SS)
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
RUN_NAME="2lvl-concat-concat"
RUN_NAME="2lvl-embed-concat"
####################

###################
### Config file ###
###################
config_file=$ACCELERATE_DIR/configs/cond_gen/cond_generator_10bit_2lvl.yaml
###################


####################
## Tokenizer ckpt ##
####################
vqgan_checkpoint=/n/holylfs06/LABS/sham_lab/Users/ydu/zhangxiangcheng/PresionAR-LFQ/maskbit/runs/outputs/rqbit_tokenizer_10bit/2level-mixed_from_scratch-long/archive/checkpoint-800000/ema_model
####################


###################
## Model args #####
###################
MODEL_ARGS="model.mlm_model.context_conditioning=embed \
    model.mlm_model.label_conditioning=concat \
    "
# MODEL_ARGS="model.mlm_model.context_conditioning=concat \
#     model.mlm_model.label_conditioning=concat \
#     model.mlm_model.tie_context_pos_embeddings=true \
#     "
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
    $ACCELERATE_DIR/scripts/train_cond_mlm.py \
    config=$config_file \
    training.per_gpu_batch_size=64 \
    training.gradient_accumulation_steps=1 \
    experiment.eval_gen_every=20_000 \
    experiment.eval_loss_every=10_000 \
    experiment.resume=true \
    experiment.run_name=${RUN_NAME} \
    experiment.vqgan_checkpoint=${vqgan_checkpoint} \
    training.mixed_precision="bf16" \
    model.mlm_model.num_steps=4 \
    model.mlm_model.depth=20 \
    model.mlm_model.hidden_dim=768 \
    model.mlm_model.heads=12 \
    ${MODEL_ARGS} \
    "


