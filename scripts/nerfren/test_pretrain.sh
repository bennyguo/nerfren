dataset=$1
python test.py --name $1_pretrain \
    --dataset_mode rffr --dataset_root ./load/rffr/$1 --img_wh 648 432 \
    --patch_size 4 --batch_size 128 \
    --model nerfren --mlp_network two_layer_mlp \
    --N_coarse 64 --N_importance 64 
    