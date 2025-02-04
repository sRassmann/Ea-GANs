# singularity container build based on flairsyn/Dockerfile
NAME=ea_gan_brats_retrain

singularity exec --nv -B /groups/ag-reuter/projects/flair_synthesis \
 "/home/$USER/$USER"_dif.sif python train.py --name $NAME --use_dropout \
 --rise_sobelLoss --batchSize 6 --config config_brats.yml --niter 150

singularity exec --nv -B /groups/ag-reuter/projects/flair_synthesis \
 "/home/$USER/$USER"_dif.sif python test.py --name $NAME --use_dropout \
 --dataset_json ../data/BraTS/brats23_train.json --data_dir ../data/BraTS/brats23_conformed \
 -o inference --no_skull_strip