# singularity container build based on flairsyn/Dockerfile
NAME=ea_gan_brats_default_lambda_1k

export APPTAINERENV_CUDA_VISIBLE_DEVICES=0

#singularity exec --nv -B /groups/ag-reuter/projects/flair_synthesis \
# "/home/$USER/$USER"_dif.sif python train.py --name $NAME --use_dropout \
# --rise_sobelLoss --batchSize 6 --config config_brats.yml --lambda_A 1000 --niter 150

singularity exec --nv -B /groups/ag-reuter/projects/flair_synthesis \
 "/home/$USER/$USER"_dif.sif python test.py --name $NAME --use_dropout \
 --dataset_json ../data/BraTS/brats23_train.json --data_dir ../data/BraTS/brats23_conformed \
 -o inference --no_skull_strip --operating_size 176 224 220
