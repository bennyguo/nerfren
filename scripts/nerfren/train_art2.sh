dataset=art2
python train.py --name $dataset \
    --dataset_mode rffr --dataset_root ./load/rffr/${dataset} --img_wh 648 432 \
    --patch_size 4 --batch_size 128 \
    --model nerfren --mlp_network two_layer_mlp \
    --N_coarse 64 --N_importance 64 \
    --lambda_trans_depth_smoothness 0.01 --lambda_refl_bdc 1e-4 --lambda_refl_depth_smoothness 1e-10 --lambda_beta_smoothness 1e-4 --lambda_trans_lowpass 0.01 \
    --lambda_schedulers "refl_bdc@step@10:0.05:12:1e-4:15:0;refl_depth_smoothness@step@15:0.01" \
    --lambda_schedulers_step "trans_depth_smoothness@step@1000:0.1:5000:0.01;trans_lowpass@step@1000:0" \
    --dir_pe_masking True --dir_pe_masking_start 9 --dir_pe_masking_end 10 
    