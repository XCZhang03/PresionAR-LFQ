#!/bin/bash

# Navigate to the maskbit directory
cd /home/linhw/zhangxiangcheng/DiffAR/PrecisionAR-LFQ/maskbit || exit

# Run the training script for the tokenizer
# PYTHONPATH=./ WORKSPACE=./ accelerate launch --num_machines=1  --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network scripts/train_rqbit_tokenizer.py config=./configs/tokenizer/rqbit_tokenizer_10bit.yaml
PYTHONPATH=./ WORKSPACE=./debug  accelerate launch --multi_gpu --num_processes 2 scripts/train_res_tokenizer.py config=./configs/tokenizer/rqgan_10bit.yaml