#!/bin/bash

#SBATCH --job-name=headnerf_cnn_40000
#SBATCH --gres=gpu:1
#SBATCH -o slurm.out
#SBATCH -p batch
#SBATCH --time=14-0
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=20G

. /data/shinahyung/anaconda3/etc/profile.d/conda.sh
conda activate adnerf

# Fitting a single image using HeadNeRF
# conda env adnerf
python FittingSingleImage.py --model_path "TrainedModels/model_Reso32HR.pth" \
                             --img "/data/shinahyung/code/LipSync_datasets/HeadNeRF_CNN_datasets/cnn_imgs/" \
                             --mask "/data/shinahyung/code/LipSync_datasets/HeadNeRF_CNN_datasets/mask_imgs/" \
                             --para_3dmm "/data/shinahyung/code/LipSync_datasets/HeadNeRF_CNN_datasets/cnn_imgs/" \
                             --save_root "./fitting_res" \
                             --target_embedding "LatentCodeSamples/*/S025_E14_I01_P02.pth"