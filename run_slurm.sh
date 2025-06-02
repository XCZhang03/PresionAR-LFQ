#!/bin/bash

#SBATCH --job-name=vae-2-from_scratch
#SBATCH -p kempner_requeue
#SBATCH --mem=100G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=504985967@qq.com
#SBATCH -o status/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e status/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --nodes=2                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --cpus-per-task=16           # number of CPU cores per task
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:4                # number of GPUs per node
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


######################
# Set work dir #######
######################
ACCELERATE_DIR="/n/holylfs06/LABS/sham_lab/Users/ydu/zhangxiangcheng/PresionAR-LFQ/maskbit"
cd $ACCELERATE_DIR
######################

####################
### Set run name ###
####################
RUN_NAME="2level-mixed_from_scratch-long"
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
    $ACCELERATE_DIR/scripts/train_res_tokenizer.py \
    config=$ACCELERATE_DIR/configs/tokenizer/rqbit_tokenizer_10bit_2lvl.yaml \
    training.per_gpu_batch_size=16 \
    training.gradient_accumulation_steps=2 \
    experiment.save_every=2_000 \
    experiment.resume=true \
    experiment.run_name=${RUN_NAME} \
    experiment.init_checkpoint=/n/holylfs06/LABS/sham_lab/Users/ydu/zhangxiangcheng/PresionAR-LFQ/maskbit/runs/outputs/rqbit_tokenizer_10bit/2level/current/checkpoints/checkpoint_105 \
    "



