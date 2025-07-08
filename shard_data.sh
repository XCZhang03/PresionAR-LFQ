#!/bin/bash
#SBATCH -c 1                # Number of cores (-c)
#SBATCH -t 0-08:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue   # Partition to submit to
#SBATCH --mem=100           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o status/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e status/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

cd /n/holylfs06/LABS/sham_lab/Users/ydu/zhangxiangcheng/PresionAR-LFQ/maskbit
source ~/miniconda3/etc/profile.d/conda.sh
conda activate maskbit
python scripts/create_sharded_dataset.py --data="./data" --shards="./shards" 