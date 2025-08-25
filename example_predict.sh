# singularity container build based on flairsyn/Dockerfile
NAME=EaGAN

# following example explained in main script

singularity exec --nv -B /groups/ag-reuter/projects/flair_synthesis \
  "/home/$USER/$USER"_dif.sif python test.py --name EaGAN --use_dropout \
  --out_dir_name inference_rs_example --dataset_json ../data/RS/rs_example.json --data_dir ../data/rs_example_conformed


