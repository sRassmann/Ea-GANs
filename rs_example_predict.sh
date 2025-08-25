# singularity container build based on flairsyn/Dockerfile
NAME=EaGAN

singularity exec --nv -B /groups/ag-reuter/projects/flair_synthesis \
  "/home/$USER/$USER"_dif.sif python test.py --name EaGAN --use_dropout \
  --out_dir_name inference_rs_example1 --dataset_json ../data/rs_example.json --data_dir ../data/rs_example_conformed --no_skull_strip


