# NeRFReN: Neural Radiance Fields with Reflections
This is the code release for our CVPR2022 paper, NeRFReN: Neural Radiance Fields with Reflections.

- [Project Page](https://bennyguo.github.io/nerfren/)
- Paper ([Conference](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_NeRFReN_Neural_Radiance_Fields_With_Reflections_CVPR_2022_paper.html), [ArXiv](https://arxiv.org/abs/2111.15234))
- [Video](https://www.youtube.com/watch?v=uZin1Ynk6SM)

## Update
- 07/28/2022: Initial code release.
- 08/01/2022: Pretrained models for all RFFR scenes are released.

## Setup
- Install PyTorch>=1.8
- Install other dependencies: `pip install -r requirements.txt`
- Download our Real Forward Facing with Reflections (RFFR) dataset from [Google Drive](https://drive.google.com/file/d/1UFHdcLgn9wXBcYZVM5qojYP1NNpg5bsS/view?usp=sharing), and extract to `load/`
- (Optional) Download pretrained models from [Google Drive](https://drive.google.com/file/d/13Im7uA2OAjSxxDm6oP4GRiC82nP1QEvM/view?usp=sharing), and extract to `checkpoints/`

The correct file structure should be like:

```
checkpoints/
  |
  -- art1_pretrain/
  |
  -- ...
load/
  |
  -- rffr/
    |
    -- art1/
    |
    -- ...
```

## Training
We provide training scripts for all the 6 RFFR scenes in `scripts/nerfren`. Run the scripts to perform training:
```{bash}
sh scripts/nerfren/train_art1.sh
```

To train the NeRF baseline, run `scripts/nerf/train.sh` and specify the scene as arguments:
```{bash}
sh scripts/nerf/train.sh art1
```

The training process by default uses all available GPUs. Set `CUDA_VISIBLE_DEVICES` environment variable to specify the GPUs to be used.

The network checkpoints and visualizations are stored in `checkpoints/` by default, and tensorboard logs can be found in `runs/`.

## Testing
The testing process generates images from spiral poses for visualization. To test a pretrained model, run `scripts/nerfren/test_pretrain.sh` and specify the scene as arguments:
```{bash}
sh scripts/nerfren/test_pretrain.sh art1
```
To test on our pretrained models, please make sure you have downloaded the checkpoints and organized the files correctly as demonstrated in the Setup section.

The testing results are saved to `results/` by default.

## Citation
If you find our work useful, please cite:
```
@InProceedings{Guo_2022_CVPR,
    author    = {Guo, Yuan-Chen and Kang, Di and Bao, Linchao and He, Yu and Zhang, Song-Hai},
    title     = {NeRFReN: Neural Radiance Fields With Reflections},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {18409-18418}
}
```

## Acknowledgement
Part of the code is borrowed or adapted from the following great codebases:
- [https://github.com/kwea123/nerf_pl](https://github.com/kwea123/nerf_pl)
- [https://github.com/google-research/google-research/tree/master/jaxnerf](https://github.com/google-research/google-research/tree/master/jaxnerf)
- [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
