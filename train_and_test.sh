# singularity container build based on flairsyn/Dockerfile
NAME=EaGAN

singularity exec --nv -B $HPCWORK "/home/$USER/$USER"_dif.sif python train.py \
  --name $NAME --use_dropout --rise_sobelLoss --batchSize 6
#
#singularity exec --nv -B $HPCWORK "/home/$USER/$USER"_dif.sif python test.py \
#  --name $NAME --use_dropout

#singularity exec --nv -B /groups/ag-reuter/projects/flair_synthesis -B $HPCWORK \
#  "/home/$USER/$USER"_dif.sif python test.py --name $NAME --use_dropout \
#  --dataset_json ../data/RS/RS_wmh_test.json -o inference_wmh --data_dir ../data/RS/conformed_test

singularity exec --nv -B /groups/ag-reuter/projects/flair_synthesis -B $HPCWORK \
  "/home/$USER/$USER"_dif.sif python test.py --name $NAME --use_dropout \
  --out_dir_name inference_t --dataset_json ../data/RS/RS_test.json --data_dir ../data/RS/conformed_test_600

singularity exec --nv -B /groups/ag-reuter/projects/flair_synthesis \
  "/home/$USER/$USER"_dif.sif python test.py --name EaGAN --use_dropout \
  --out_dir_name inference_bmb --dataset_json ../data/test_datasets/bmb_bbreg.json --data_dir ../data/test_datasets/bmb

#singularity exec --nv -B /groups/ag-reuter/projects/flair_synthesis -B $HPCWORK \
#  "/home/$USER/$USER"_dif.sif python test.py --name $NAME --use_dropout \
#  --dataset_json ../data/RS/RS_pvs_test.json -o inference_pvs --data_dir ../data/RS/conformed_pvs_test
#
### downstream
#cd ../SHIVA_WMH/ && sh run_predict_from_out_dir.sh ../flairsyn/output/Ea-GAN/inference_wmh pred_flair_n4.nii.gz \
#cd ../SHIVA_PVS/ && sh run_predict_from_out_dir.sh ../flairsyn/output/Ea-GAN/inference_pvs pred_flair_n4.nii.gz && \
# cd ../flairsyn && python metrics_lesions.py output/Ea-GAN/inference_wmh --pv_sep --pred_file_name pred_flair_n4_wmh_seg.nii.gz && \
# cd ../SHIVA_WMH/ && sh run_predict_from_out_dir.sh ../flairsyn/output/Ea-GAN/inference_t pred_flair_n4.nii.gz
#
#python python scripts/postprocessing/n4_bias_field_correction.py output/Ea-GAN/inference_t && python metrics_lesions.py output/Ea-GAN/inference_pvs -pvs --pred_file_name pred_flair_n4_pvs_seg.nii.gz && \
#
