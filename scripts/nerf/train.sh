dataset=$1
python train.py --name $1_nerf \
    --dataset_mode rffr --dataset_root ./load/rffr/$1 --img_wh 648 432 \
    --patch_size 1 --batch_size 2048 \
    --model nerf --mlp_network vanilla_mlp \
    --N_coarse 64 --N_importance 64 
    