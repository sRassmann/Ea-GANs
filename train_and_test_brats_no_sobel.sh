#!/bin/bash
#SBATCH --job-name=ea_gan_noSob
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=2-00:10:00
#SBATCH --partition=HPC-4GPUs
#SBATCH --gres=gpu:1
#SBATCH --output=/home/rassmanns/diffusion/Ea-GANs/output/logs/ea_gan_noSob.out

module load singularity
cd $HOME/diffusion/Ea-GANs

# singularity container build based on flairsyn/Dockerfile
NAME=ea_gan_brats_no_sobel

export APPTAINERENV_CUDA_VISIBLE_DEVICES=0

singularity exec --nv -B /groups/ag-reuter/projects/flair_synthesis \
 "/home/$USER/$USER"_dif.sif python train.py --name $NAME --use_dropout \
 --batchSize 6 --config config_brats.yml --lambda_A 300 --niter 150 \
 --lambda_sobel 0

singularity exec --nv -B /groups/ag-reuter/projects/flair_synthesis \
 "/home/$USER/$USER"_dif.sif python test.py --name $NAME --use_dropout \
 --dataset_json ../data/BraTS/brats23_train.json --data_dir ../data/BraTS/brats23_conformed \
 -o inference --no_skull_strip --operating_size 176 224 220
