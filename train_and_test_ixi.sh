# singularity container build based on flairsyn/Dockerfile
NAME=ea_gan_ixi_no_skstrp

SINGULARITYENV_CUDA_VISIBLE_DEVICES=1 singularity exec --nv -B /groups/ag-reuter/projects/flair_synthesis  \
 "/home/$USER/$USER"_dif.sif python train.py --name $NAME --use_dropout \
 --rise_sobelLoss --batchSize 6 --config config_ixi.yml

SINGULARITYENV_CUDA_VISIBLE_DEVICES=1 singularity exec --nv -B /groups/ag-reuter/projects/flair_synthesis \
 "/home/$USER/$USER"_dif.sif python test.py --name $NAME --use_dropout \
 --dataset_json ../data/test_datasets/ixi_train.json --data_dir ../data/test_datasets/ixi \
 -o inference --config config_ixi.yml --no_skull_strip --operating_size 176 224 224