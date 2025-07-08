###################
### Conda Env #####
###################
source /n/holylabs/ydu_lab/Lab/zhangxiangcheng/miniconda3/etc/profile.d/conda.sh
conda activate maskbit
###################

######################
# Set work dir #######
######################
export ACCELERATE_DIR="/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/PresionAR-LFQ/maskbit"
# export ACCELERATE_DIR="/datapool/data2/home/linhw/zhangxiangcheng/DiffAR/PrecisionAR-LFQ/maskbit"
cd $ACCELERATE_DIR
######################


####################
### Set wandb ######
export WANDB_API_KEY=78319f33ffd79b3480286266fd8ba1f9c5bc3dab
export WANDB_ENTITY=DiffAR
####################


