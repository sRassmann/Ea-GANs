# singularity container build based on flairsyn/Dockerfile
NAME=EaGAN

singularity exec --nv -B $HPCWORK "/home/$USER/$USER"_dif.sif python train.py \
  --name $NAME --use_dropout --rise_sobelLoss --batchSize 6

singularity exec --nv -B $HPCWORK "/home/$USER/$USER"_dif.sif python test.py \
  --name $NAME --use_dropout

singularity exec --nv -B /groups/ag-reuter/projects/flair_synthesis -B $HPCWORK \
  "/home/$USER/$USER"_dif.sif python test.py --name $NAME --use_dropout \
  --dataset_json ../data/RS/RS_wmh_test.json -o inference_wmh --data_dir ../data/RS/conformed_test

singularity exec --nv -B /groups/ag-reuter/projects/flair_synthesis -B $HPCWORK \
  "/home/$USER/$USER"_dif.sif python test.py --name $NAME --use_dropout \
  --dataset_json ../data/RS/RS_pvs_test.json -o inference_pvs --data_dir ../data/RS/conformed_pvs_test

singularity exec --nv -B /groups/ag-reuter/projects/flair_synthesis -B $HPCWORK \
  "/home/$USER/$USER"_dif.sif python test.py --name $NAME --use_dropout \
  --out_dir_name inference_t --dataset_json ../data/RS/RS_test.json --data_dir ../data/RS/conformed_test_600


