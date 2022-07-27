"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import bisect
from tqdm import tqdm
import itertools
import torch
import torch.nn as nn

from models import find_network_using_name
from options import get_option_setter, str2bool
from utils.utils import chunk_batch, find_class_using_name
from utils.visualizer import VisualizeItem, depth2im
from .base_model import BaseModel
from .networks import init_net
from .embedding import BaseEmbedding
from .rendering import VolumetricRenderer
from .utils import *
from .criterions import *


class NeRFModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--mlp_network', type=str, default='vanilla_mlp')
        parser.add_argument('--embedding', type=str, default='positional_encoding')
        parser.add_argument('--N_coarse', type=int, default=64)
        parser.add_argument('--N_importance', type=int, default=64)
        parser.add_argument('--noise_std', type=float, default=0., help="std dev of noise added to regularize sigma output. (used in the llff dataset only)")
        parser.add_argument('--white_bkgd', action='store_true', help="using white color as default background. (used in the blender dataset only)")
        parser.add_argument('--randomized', type=str2bool, default=True)

        parser.add_argument('--dim_rgb', type=int, default=3)
        parser.add_argument('--dim_pos', type=int, default=3)
        parser.add_argument('--dim_dir', type=int, default=3)
        parser.add_argument('--deg_pos', type=int, default=10)
        parser.add_argument('--deg_dir', type=int, default=4)

        parser.add_argument('--lambda_coarse_mse', type=float, default=1.)
        parser.add_argument('--lambda_fine_mse', type=float, default=1.)
        parser.add_argument('--lambda_coarse_depth_smoothness', type=float, default=0.)
        parser.add_argument('--lambda_fine_depth_smoothness', type=float, default=0.)

        """
        lambda scheduler format:
        [name]@[step/linear/exp]@[epoch1]:[lr1]:[epoch2]:[lr2]:..:[epochN]:[lrN];...
        """
        parser.add_argument('--lambda_schedulers', type=str, default='')
        parser.add_argument('--lambda_schedulers_step', type=str, default='')        

        parser.add_argument('--color_activation', type=str, default='sigmoid', choices=['none', 'relu', 'sigmoid'])
        parser.add_argument('--sigma_activation', type=str, default='relu', choices=['relu', 'softplus'])

        parser.add_argument('--bilateral_gamma', type=float, default=0.1)
        parser.add_argument('--save_data', type=str2bool, default=False)

        opt, _ = parser.parse_known_args()
        embedding_option_setter = get_option_setter(find_class_using_name('models.embedding', opt.embedding, type=BaseEmbedding))
        parser = embedding_option_setter(parser)

        opt, _ = parser.parse_known_args()
        for key, network_name in opt.__dict__.items():
            if key.endswith('_network'):
                network_option_setter = get_option_setter(find_class_using_name('models.networks', network_name, type=nn.Module))
                parser = network_option_setter(parser)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel

        # specify the losses you want to print out.
        self.train_loss_names = ['coarse_mse', 'fine_mse', 'coarse_psnr', 'fine_psnr', 'tot']
        self.val_iter_loss_names = ['coarse_mse', 'fine_mse', 'coarse_psnr', 'fine_psnr', 'tot']
        self.val_loss_names = ['coarse_psnr', 'fine_psnr']
        self.test_loss_names = []

        if opt.lambda_coarse_depth_smoothness > 0 and opt.patch_size > 2:
            self.train_loss_names += ['coarse_depth_smoothness']
        if opt.lambda_fine_depth_smoothness > 0 and opt.patch_size > 2:
            self.train_loss_names += ['fine_depth_smoothness']         

        # specify the data you want to save.
        self.train_visual_names = []
        self.val_iter_visual_names = ['coarse_pred', 'fine_pred']
        self.val_visual_names = ['coarse_pred', 'fine_pred']
        self.test_visual_names = ['coarse_pred', 'fine_pred', 'coarse_pred_gif', 'fine_pred_gif']
        if opt.save_data:
            self.val_visual_names += ['coarse_data', 'fine_data']
            self.test_visual_names += ['coarse_data', 'fine_data']

        # specify the models you want to save to the disk.
        self.model_names = ['Coarse', 'Fine']

        self.netCoarse = init_net(find_network_using_name(opt.mlp_network)(opt), opt)
        self.netFine = init_net(find_network_using_name(opt.mlp_network)(opt), opt)
        self.embeddings = {
            'pos': find_class_using_name('models.embedding', opt.embedding, type=BaseEmbedding)(opt.dim_pos, opt.deg_pos),
            'dir': find_class_using_name('models.embedding', opt.embedding, type=BaseEmbedding)(opt.dim_dir, opt.deg_dir)
        }
        self.models = {
            'coarse': self.netCoarse,
            'fine': self.netFine
        }
        self.losses = {
            'mse': MSELoss(opt),
            'psnr': PSNR(opt),
            'edge-preserving-smoothness': EdgePreservingSmoothnessLoss(opt)
        }
        self.renderer = VolumetricRenderer(self.opt)
        self.randomized = opt.randomized
        if self.isTrain:  # only defined during training time
            # define and initialize optimizers. You can define one optimizer for each network.
            self.optimizer = torch.optim.Adam([{
                'params': itertools.chain(self.netCoarse.parameters(), self.netFine.parameters()),
                'initial_lr': opt.lr
            }], lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer]

        self.lambda_schedulers, self.lambda_schedulers_step = self.parse_lambda_schedulers(self.opt.lambda_schedulers), self.parse_lambda_schedulers(self.opt.lambda_schedulers_step)

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        x.ndim == 3 when
        1. [train with patch_size > 1] (N, P**2, D)
        2. [val/test] (1, H*W,D)
        pack into (N_rays, D)
        """
        pack = lambda x: x.view(-1, x.shape[-1]) if x.ndim == 3 else x
        for name, v in input.items():
            setattr(self, f"data_{name}", pack(v).to(self.device))
    
    def train(self):
        super().train()
        self.randomized = self.opt.randomized
        self.H, self.W = self.opt.patch_size, self.opt.patch_size

    def eval(self):
        super().eval()
        self.randomized = False
        self.H, self.W = self.opt.img_wh[1], self.opt.img_wh[0]

    def parse_lambda_schedulers(self, schedulers_str):
        # [name]@[step/linear/exp]@[epoch1]:[lr1]:[epoch2]:[lr2]:..:[epochN]:[lrN];...
        # step: [epoch1, epoch2)
        schedulers_str = schedulers_str.split(';')
        schedulers = {}
        step_inf = int(1e10)
        for s in filter(None, schedulers_str):
            name, sched_type, params = s.split('@')
            assert sched_type in ['step', 'linear']
            params = params.split(':')
            steps, lr = list(map(int, params[0::2])), list(map(float, params[1::2]))
            schedulers[name] = {
                'type': sched_type,
                'steps': [1] + steps + [step_inf],
                'lr': [getattr(self.opt, 'lambda_' + name)] + lr + [lr[-1]]
            }
        return schedulers
    
    def update_hyperparameters(self, epoch):
        super().update_hyperparameters(epoch)
        for name, s in self.lambda_schedulers.items():
            start = bisect.bisect_right(s['steps'], epoch) - 1
            step_start, step_end = s['steps'][start], s['steps'][start + 1]
            lr_start, lr_end = s['lr'][start], s['lr'][start + 1]
            if s['type'] == 'step':
                lr = lr_start
            elif s['type'] == 'linear':
                lr = lr_start + (epoch - step_start) / (step_end - step_start) * (lr_end - lr_start)
            setattr(self.opt, 'lambda_' + name, lr)
            print(f"Set hyperparameter lambda_{name} to {lr:.3e}")

    def update_hyperparameters_step(self, step):
        super().update_hyperparameters_step(step)
        for name, s in self.lambda_schedulers_step.items():
            start = bisect.bisect_right(s['steps'], step) - 1
            step_start, step_end = s['steps'][start], s['steps'][start + 1]
            lr_start, lr_end = s['lr'][start], s['lr'][start + 1]
            if s['type'] == 'step':
                lr = lr_start
            elif s['type'] == 'linear':
                lr = lr_start + (step - step_start) / (step_end - step_start) * (lr_end - lr_start)
            setattr(self.opt, 'lambda_' + name, lr)        

    def render_rays(self, model, xyz, dir_embedded, **kwargs):
        N_rays = xyz.shape[0]
        N_samples = xyz.shape[1]
        xyz = xyz.view(-1, self.opt.dim_pos)
        B = xyz.shape[0]
        xyz_embedded = self.embeddings['pos'](xyz)
        """
        THIS IS WRONG:
        dir_embedded = dir_embedded.repeat(N_samples, 1)
        THESE ARE RIGHT:
        dir_embedded = repeat(dir_embedded, 'n1 c -> (n1 n2) c', n2=N_samples)
        dir_embedded = dir_embedded[:,None,...].expand(-1, N_samples, -1).reshape(N_rays * N_samples, -1)
        """
        dir_embedded = dir_embedded[:,None,...].expand(-1, N_samples, -1).reshape(N_rays * N_samples, -1)
        xyzdir_embedded = torch.cat([xyz_embedded, dir_embedded], dim=-1)
        out = chunk_batch(model, self.opt.point_chunk, xyzdir_embedded, **kwargs)
        out = out.view(N_rays, N_samples, -1)
        out_rgbs = out[..., :self.opt.dim_rgb]
        out_sigmas = out[..., self.opt.dim_rgb]
        return out_rgbs, out_sigmas

    def forward_rays(self, rays):
        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
        near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)
        dir_embedded = self.embeddings['dir'](rays[:, 8:11]) # (N_rays, embed_dir_channels)
        z_vals, xyz_coarse = sample_along_rays(rays_o, rays_d, near, far, self.opt.N_coarse, self.randomized, lindisp=False)

        coarse_rgbs, coarse_sigmas = self.render_rays(self.models['coarse'], xyz_coarse, dir_embedded)
        coarse_sigmas = add_gaussian_noise(coarse_sigmas, self.randomized, self.opt.noise_std)
        coarse_comp_rgbs, coarse_depth, coarse_opacity, coarse_weights = self.renderer(coarse_rgbs, coarse_sigmas, z_vals, self.opt.white_bkgd)

        out = {
            'coarse_comp_rgbs': coarse_comp_rgbs,
            'coarse_depth': coarse_depth,
            'coarse_opacity': coarse_opacity,
            'coarse_weights': coarse_weights,
            'coarse_sigmas': coarse_sigmas,
            'coarse_z_samples': z_vals
        }

        if self.opt.N_importance > 0: # sample points for fine model
            z_vals, xyz_fine = resample_along_rays(rays_o, rays_d, z_vals, coarse_weights.detach(), self.opt.N_importance, self.randomized)
            fine_rgbs, fine_sigmas = self.render_rays(self.models['fine'], xyz_fine, dir_embedded)
            fine_sigmas = add_gaussian_noise(fine_sigmas, self.randomized, self.opt.noise_std)
            fine_comp_rgbs, fine_depth, fine_opacity, fine_weights = self.renderer(fine_rgbs, fine_sigmas, z_vals, self.opt.white_bkgd)
            out.update({
                'fine_comp_rgbs': fine_comp_rgbs,
                'fine_depth': fine_depth,
                'fine_opacity': fine_opacity,
                'fine_weights': fine_weights,
                'fine_sigmas': fine_sigmas,
                'fine_z_samples': z_vals
            })

        return out


    def forward(self):
        out = chunk_batch(self.forward_rays, self.opt.ray_chunk, self.data_rays)
        for name, v in out.items():
            setattr(self, f"out_{name}", v)
    
    def calculate_losses(self):
        self.loss_tot = 0

        # coarse mse
        self.loss_coarse_mse = self.losses['mse'](self.out_coarse_comp_rgbs, self.data_rgbs) * self.opt.lambda_coarse_mse
        self.loss_tot += self.loss_coarse_mse

        # fine mse
        if hasattr(self, 'out_fine_comp_rgbs'):
            self.loss_fine_mse = self.losses['mse'](self.out_fine_comp_rgbs, self.data_rgbs) * self.opt.lambda_fine_mse
        else:
            self.loss_fine_mse = 0
        self.loss_tot += self.loss_fine_mse

        # coarse depth smoothness
        if self.opt.lambda_coarse_depth_smoothness > 0 and self.opt.patch_size > 2:
            self.loss_coarse_depth_smoothness = self.losses['edge-preserving-smoothness'](
                self.out_coarse_depth.view(-1, self.H, self.W),
                self.data_rgbs.view(-1, self.H, self.W, self.opt.dim_rgb)
            ) * self.opt.lambda_coarse_depth_smoothness
            self.loss_tot += self.loss_coarse_depth_smoothness

        # fine depth smoothness
        if self.opt.lambda_fine_depth_smoothness > 0 and self.opt.patch_size > 2:
            if hasattr(self, 'out_fine_depth'):
                self.loss_fine_depth_lap = self.losses['edge-preserving-smoothness'](
                    self.out_fine_depth.view(-1, self.H, self.W),
                    self.data_rgbs.view(-1, self.H, self.W, self.opt.dim_rgb)
                ) * self.opt.lambda_fine_depth_smoothness
                self.loss_tot += self.loss_fine_depth_smoothness         

        with torch.no_grad():
            self.loss_coarse_psnr = self.losses['psnr'](self.out_coarse_comp_rgbs, self.data_rgbs)
            if hasattr(self, 'out_fine_comp_rgbs'):
                self.loss_fine_psnr = self.losses['psnr'](self.out_fine_comp_rgbs, self.data_rgbs)
            else:
                self.loss_fine_psnr = 0

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.calculate_losses()
        self.loss_tot.backward()

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        if self.opt.grad_clip_val > 0:
            if self.opt.grad_clip_type == 'norm':
                nn.utils.clip_grad_norm_(itertools.chain(self.netCoarse.parameters(), self.netFine.parameters()), self.opt.grad_clip_val)
            elif self.opt.grad_clip_type == 'value':
                nn.utils.clip_grad_value_(itertools.chain(self.netCoarse.parameters(), self.netFine.parameters()), self.opt.grad_clip_val)
        self.optimizer.step()

    def calculate_vis(self, with_gt):
        W, H = self.opt.img_wh
        coarse_img = self.out_coarse_comp_rgbs.view(H, W, 3).cpu()
        coarse_depth = depth2im(self.out_coarse_depth.view(H, W), opt=self.opt)
        self.coarse_data_dict = {
            'comp_rgbs': self.out_coarse_comp_rgbs.view(H, W, -1),
            'depth': self.out_coarse_depth.view(H, W),
            'sigmas': self.out_coarse_sigmas.view(H, W, -1),
            'z_vals': self.out_coarse_z_samples.view(H, W, -1)
        }
        if self.opt.N_importance > 0:
            fine_img = self.out_fine_comp_rgbs.view(H, W, 3).cpu()
            fine_depth = depth2im(self.out_fine_depth.view(H, W), opt=self.opt)
            self.fine_data_dict = {
                'comp_rgbs': self.out_fine_comp_rgbs.view(H, W, -1),
                'depth': self.out_fine_depth.view(H, W),
                'sigmas': self.out_fine_sigmas.view(H, W, -1),
                'z_vals': self.out_fine_z_samples.view(H, W, -1)
            }
        if with_gt:
            gt_img = self.data_rgbs.view(H, W, 3).cpu()
            self.coarse_data_dict['gt_rgbs'] = gt_img
            coarse_pred = torch.cat([gt_img, coarse_img, coarse_depth], dim=1)
            self.coarse_pred_img = coarse_pred
            if self.opt.N_importance > 0:
                self.fine_data_dict['gt_rgbs'] = gt_img
                fine_pred = torch.cat([gt_img, fine_img, fine_depth], dim=1)
                self.fine_pred_img = fine_pred
        else:
            coarse_pred = torch.cat([coarse_img, coarse_depth], dim=1)
            self.coarse_pred_img = coarse_pred
            if self.opt.N_importance > 0:
                fine_pred = torch.cat([fine_img, fine_depth], dim=1)
                self.fine_pred_img = fine_pred

    def validate_iter(self):
        self.forward()
        self.calculate_losses()
        self.calculate_vis(with_gt=True)
        self.coarse_pred = VisualizeItem('image', self.coarse_pred_img, timestamp=True, name='coarse', data_format='HWC', range=(0, 1), img_format='png')
        if self.opt.N_importance > 0:
            self.fine_pred = VisualizeItem('image', self.fine_pred_img, timestamp=True, name='fine', data_format='HWC', range=(0, 1), img_format='png')
            
    def validate(self, dataset):
        coarse_psnr, fine_psnr = 0, 0
        coarse_pred, fine_pred = [], []
        coarse_data, fine_data = [], []
        for i, data in enumerate(tqdm(dataset, desc="Validation", total=len(dataset.dataloader))):
            self.set_input(data)
            self.forward()
            self.calculate_losses()
            coarse_psnr += self.loss_coarse_psnr
            fine_psnr += self.loss_fine_psnr
            self.calculate_vis(with_gt=True)
            coarse_pred.append(
                VisualizeItem('image', self.coarse_pred_img, timestamp=False, name=f'{i}-coarse', data_format='HWC', range=(0, 1), img_format='png')
            )
            coarse_data.append(
                VisualizeItem('data', self.coarse_data_dict, timestamp=False, name=f'{i}-coarse')
            )
            if self.opt.N_importance > 0:
                fine_pred.append(
                    VisualizeItem('image', self.fine_pred_img, timestamp=False, name=f'{i}-fine', data_format='HWC', range=(0, 1), img_format='png')
                )
                fine_data.append(
                    VisualizeItem('data', self.fine_data_dict, timestamp=False, name=f'{i}-fine')
                )
        self.loss_coarse_psnr = coarse_psnr / len(dataset)
        self.loss_fine_psnr = fine_psnr / len(dataset)
        self.coarse_pred = coarse_pred
        self.coarse_data = coarse_data
        if self.opt.N_importance > 0:
            self.fine_pred = fine_pred
            self.fine_data = fine_data

    def test(self, dataset):
        coarse_pred_imgs, fine_pred_imgs = [], []
        self.coarse_data, self.fine_data = [], []
        for i, data in enumerate(tqdm(dataset, desc="Testing", total=len(dataset.dataloader))):
            self.set_input(data)
            self.forward()
            self.calculate_vis(with_gt=False)
            coarse_pred_imgs.append(self.coarse_pred_img)
            self.coarse_data.append(
                VisualizeItem('data', self.coarse_data_dict, timestamp=False, name=f'{i}-coarse')
            )
            if self.opt.N_importance > 0:
                fine_pred_imgs.append(self.fine_pred_img)
                self.fine_data.append(
                    VisualizeItem('data', self.fine_data_dict, timestamp=False, name=f'{i}-fine')
                )
        self.coarse_pred, self.fine_pred = [], []
        if self.opt.N_importance > 0:
            for i, (coarse_pred_img, fine_pred_img) in enumerate(zip(coarse_pred_imgs, fine_pred_imgs)):
                self.coarse_pred.append(
                    VisualizeItem('image', coarse_pred_img, timestamp=False, name=f'{i}-coarse', data_format='HWC', range=(0, 1), img_format='png')
                )
                self.fine_pred.append(
                    VisualizeItem('image', fine_pred_img, timestamp=False, name=f'{i}-fine', data_format='HWC', range=(0, 1), img_format='png')
                )
        else:
            for i, coarse_pred_img in enumerate(coarse_pred_imgs):
                self.coarse_pred.append(
                    VisualizeItem('image', coarse_pred_img, timestamp=False, name=f'{i}-coarse', data_format='HWC', range=(0, 1), img_format='png')
                )
        self.coarse_pred_gif = VisualizeItem('gif', coarse_pred_imgs, timestamp=False, name=f'coarse', data_format='HWC', range=(0, 1))
        if self.opt.N_importance > 0:
            self.fine_pred_gif = VisualizeItem('gif', fine_pred_imgs, timestamp=False, name=f'fine', data_format='HWC', range=(0, 1))
