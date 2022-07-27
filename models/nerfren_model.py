import math
import torch
import torch.nn as nn
import torch.nn.functional as TF
import itertools

from models import find_network_using_name
from options import get_option_setter, str2bool
from utils.utils import chunk_batch, find_class_using_name
from utils.visualizer import depth2im
from .base_model import BaseModel
from .networks import init_net
from .embedding import BaseEmbedding
from .rendering import VolumetricRenderer
from .utils import *
from .criterions import *
from .nerf_model import NeRFModel


class NeRFReNModel(NeRFModel):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--mlp_network', type=str, default='two_layer_mlp')
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

        parser.add_argument('--lambda_mse', type=float, default=1.)
        parser.add_argument('--lambda_trans_depth_smoothness', type=float, default=0.)
        parser.add_argument('--lambda_trans_lowpass', type=float, default=0.)
        parser.add_argument('--lambda_refl_depth_smoothness', type=float, default=0.)
        parser.add_argument('--lambda_refl_bdc', type=float, default=0.)
        parser.add_argument('--lambda_beta_smoothness', type=float, default=0.)
        parser.add_argument('--lambda_beta_mask', type=float, default=0.)
        parser.add_argument('--lambda_coarse_fine_align', type=float, default=0.)

        """
        lambda scheduler format:
        [name]@[step/linear/exp]@[epoch1]:[lr1]:[epoch2]:[lr2]:..:[epochN]:[lrN];...
        """
        parser.add_argument('--lambda_schedulers', type=str, default='')
        parser.add_argument('--lambda_schedulers_step', type=str, default='')

        parser.add_argument('--color_activation', type=str, default='sigmoid', choices=['none', 'relu', 'sigmoid'])
        parser.add_argument('--sigma_activation', type=str, default='relu', choices=['relu', 'softplus'])
        parser.add_argument('--beta_activation', type=str, default='sigmoid', choices=['none', 'sigmoid'])

        parser.add_argument('--bilateral_gamma', type=float, default=0.1)

        parser.add_argument('--with_mask', action='store_true')
        
        parser.add_argument('--dir_pe_masking', type=str2bool, default=False)
        parser.add_argument('--dir_pe_masking_start', type=int, default=10)
        parser.add_argument('--dir_pe_masking_end', type=int, default=20)

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
        BaseModel.__init__(self, opt)

        self.train_loss_names = ['coarse_mse', 'fine_mse', 'coarse_psnr', 'fine_psnr', 'tot']
        self.val_iter_loss_names = ['coarse_mse', 'fine_mse', 'coarse_psnr', 'fine_psnr', 'tot']
        self.val_loss_names = ['coarse_psnr', 'fine_psnr']
        self.test_loss_names = []

        if opt.lambda_trans_depth_smoothness > 0 and opt.patch_size >= 2:
            self.train_loss_names += ['coarse_trans_depth_smoothness', 'fine_trans_depth_smoothness']
        
        if self.opt.lambda_refl_depth_smoothness > 0 and self.opt.patch_size > 2:
            self.train_loss_names += ['coarse_refl_depth_smoothness', 'fine_refl_depth_smoothness']
        
        if opt.lambda_beta_smoothness > 0 and opt.patch_size > 2:
            self.train_loss_names += ['coarse_beta_smoothness', 'fine_beta_smoothness']
        
        if opt.lambda_beta_mask > 0 and opt.with_mask:
            self.train_loss_names += ['coarse_beta_mask', 'fine_beta_mask']
        
        if opt.lambda_trans_lowpass > 0 and opt.patch_size > 2:
            self.train_loss_names += ['coarse_trans_lowpass', 'fine_trans_lowpass']
        
        if opt.lambda_refl_bdc > 0:
            self.train_loss_names += ['coarse_refl_bdc', 'fine_refl_bdc']
        
        if opt.lambda_coarse_fine_align > 0:
            self.train_loss_names += ['coarse_fine_align']

        self.train_visual_names = []
        self.val_iter_visual_names = ['coarse_pred', 'fine_pred']
        self.val_visual_names = ['coarse_pred', 'fine_pred', 'coarse_depth', 'fine_depth']
        self.test_visual_names = ['coarse_pred', 'fine_pred', 'coarse_pred_gif', 'fine_pred_gif', 'coarse_depth', 'fine_depth']
        if opt.save_data:
            self.val_visual_names += ['coarse_data', 'fine_data']
            self.test_visual_names += ['coarse_data', 'fine_data']

        self.model_names = ['Coarse']
        if opt.N_importance > 0:
            self.model_names += ['Fine']

        self.netCoarse = init_net(find_network_using_name(opt.mlp_network)(opt), opt)
        if opt.N_importance > 0:
            self.netFine = init_net(find_network_using_name(opt.mlp_network)(opt), opt)
        self.embeddings = {
            'pos': find_class_using_name('models.embedding', opt.embedding, type=BaseEmbedding)(opt.dim_pos, opt.deg_pos),
            'dir': find_class_using_name('models.embedding', opt.embedding, type=BaseEmbedding)(opt.dim_dir, opt.deg_dir, no_xyz=True)
        }
        self.dir_pe_mask = torch.ones((self.embeddings['dir'].out_channels), dtype=torch.float32).to(self.device)
        self.models = {
            'coarse': self.netCoarse,
        }
        if opt.N_importance > 0:
            self.models['fine'] = self.netFine
        self.losses = {
            'mse': MSELoss(opt),
            'l1': L1Loss(opt),
            'psnr': PSNR(opt),
            'smoothness': SmoothnessLoss(opt),            
            'edge-preserving-smoothness': EdgePreservingSmoothnessLoss(opt),
        }
        self.renderer = VolumetricRenderer(self.opt)
        self.render_func = self.renderer.forward_bidir
        self.randomized = opt.randomized
        if self.isTrain:
            self.optimizer = torch.optim.Adam([{
                'params': itertools.chain(self.netCoarse.parameters(), self.netFine.parameters()) if opt.N_importance > 0 else self.netCoarse.parameters(),
                'initial_lr': opt.lr
            }], lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer]
        self.lambda_schedulers, self.lambda_schedulers_step = self.parse_lambda_schedulers(self.opt.lambda_schedulers), self.parse_lambda_schedulers(self.opt.lambda_schedulers_step)
    
    def update_hyperparameters(self, epoch):
        super().update_hyperparameters(epoch)
        if self.opt.dir_pe_masking:
            self.dir_pe_mask = (1 - torch.cos(math.pi * ((epoch - self.opt.dir_pe_masking_start) / (self.opt.dir_pe_masking_end - self.opt.dir_pe_masking_start) * self.opt.deg_dir - torch.arange(0, self.opt.deg_dir)).clamp(0, 1))) / 2
            self.dir_pe_mask = self.dir_pe_mask.repeat_interleave(len(self.embeddings['dir'].funcs) * self.embeddings['dir'].in_channels)
            self.dir_pe_mask = self.dir_pe_mask.to(self.device)

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
        out = {k: v.view(N_rays, N_samples, -1) if v.shape[-1] > 1 else v.view(N_rays, N_samples) for k, v in out.items()}
        if 'beta' in out:
            out['beta'] = out['beta'].view(N_rays, N_samples, 1) # beta need to be [N_rays, N_samples, 1]
        return out

    def forward_rays(self, rays):
        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
        near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)
        dir_embedded = self.embeddings['dir'](rays[:, 8:11]) # (N_rays, embed_dir_channels)
        if self.opt.dir_pe_masking:
            dir_embedded = dir_embedded * self.dir_pe_mask
        z_vals, xyz_coarse = sample_along_rays(rays_o, rays_d, near, far, self.opt.N_coarse, self.randomized, lindisp=False)

        coarse_net_out = self.render_rays(self.models['coarse'], xyz_coarse, dir_embedded)

        coarse_trans_sigmas = add_gaussian_noise(coarse_net_out['trans_sigma'], self.randomized, self.opt.noise_std)
        coarse_refl_sigmas = add_gaussian_noise(coarse_net_out['refl_sigma'], self.randomized, self.opt.noise_std)
        coarse_trans_comp_rgbs, coarse_refl_comp_rgbs, coarse_comp_rgbs, coarse_trans_depth, coarse_refl_depth, coarse_trans_depth_b2f, coarse_refl_depth_b2f, coarse_comp_beta, coarse_trans_sigmas, coarse_refl_sigmas, coarse_trans_opacity, coarse_refl_opacity, coarse_trans_weights, coarse_refl_weights = self.render_func(
            coarse_net_out['trans_rgb'], coarse_trans_sigmas, coarse_net_out['refl_rgb'], coarse_refl_sigmas, coarse_net_out['beta'], z_vals, self.opt.white_bkgd
        )

        out = {
            'coarse_trans_comp_rgbs': coarse_trans_comp_rgbs,
            'coarse_trans_depth': coarse_trans_depth,
            'coarse_trans_depth_b2f': coarse_trans_depth_b2f,
            'coarse_refl_comp_rgbs': coarse_refl_comp_rgbs,
            'coarse_refl_depth': coarse_refl_depth,
            'coarse_refl_depth_b2f': coarse_refl_depth_b2f,
            'coarse_comp_rgbs': coarse_comp_rgbs,
            'coarse_comp_beta': coarse_comp_beta,
            'coarse_trans_sigmas': coarse_trans_sigmas,
            'coarse_z_vals': z_vals,
            'coarse_refl_sigmas': coarse_refl_sigmas,
            'coarse_trans_opacity': coarse_trans_opacity,
            'coarse_refl_opacity': coarse_refl_opacity,
            'coarse_trans_weights': coarse_trans_weights,
            'coarse_refl_weights': coarse_refl_weights,
        }

        if self.opt.N_importance > 0:
            trans_z_vals, trans_xyz_fine = resample_along_rays(rays_o, rays_d, z_vals, coarse_trans_weights.detach(), self.opt.N_importance, self.randomized)
            refl_z_vals, refl_xyz_fine = resample_along_rays(rays_o, rays_d, z_vals, coarse_refl_weights.detach(), self.opt.N_importance, self.randomized)
            trans_fine_net_out = self.render_rays(self.models['fine'], trans_xyz_fine, dir_embedded, return_type='trans')
            refl_fine_net_out = self.render_rays(self.models['fine'], refl_xyz_fine, dir_embedded, return_type='refl')
            fine_trans_sigmas = add_gaussian_noise(trans_fine_net_out['trans_sigma'], self.randomized, self.opt.noise_std)
            fine_refl_sigmas = add_gaussian_noise(refl_fine_net_out['refl_sigma'], self.randomized, self.opt.noise_std)
            fine_trans_comp_rgbs, fine_refl_comp_rgbs, fine_comp_rgbs, fine_trans_depth, fine_refl_depth, fine_trans_depth_b2f, fine_refl_depth_b2f, fine_comp_beta, fine_trans_sigmas, fine_refl_sigmas, fine_trans_opacity, fine_refl_opacity, fine_trans_weights, fine_refl_weights = self.render_func(
                trans_fine_net_out['trans_rgb'], fine_trans_sigmas, refl_fine_net_out['refl_rgb'], fine_refl_sigmas, trans_fine_net_out['beta'], (trans_z_vals, refl_z_vals), self.opt.white_bkgd
            )
            out.update({
                'fine_trans_comp_rgbs': fine_trans_comp_rgbs,
                'fine_trans_depth': fine_trans_depth,
                'fine_trans_depth_b2f': fine_trans_depth_b2f,
                'fine_refl_comp_rgbs': fine_refl_comp_rgbs,
                'fine_refl_depth': fine_refl_depth,
                'fine_refl_depth_b2f': fine_refl_depth_b2f,
                'fine_comp_rgbs': fine_comp_rgbs,
                'fine_comp_beta': fine_comp_beta,
                'fine_trans_sigmas': fine_trans_sigmas,
                'fine_refl_sigmas': fine_refl_sigmas,
                'fine_trans_opacity': fine_trans_opacity,
                'fine_refl_opacity': fine_refl_opacity,
                'fine_trans_weights': fine_trans_weights,
                'fine_refl_weights': fine_refl_weights,
            })

        return out
    
    def calculate_losses(self):
        self.loss_tot = 0

        # mse loss
        self.loss_coarse_mse = self.losses['mse'](self.out_coarse_comp_rgbs, self.data_rgbs) * self.opt.lambda_mse
        self.loss_tot += self.loss_coarse_mse

        if hasattr(self, 'out_fine_comp_rgbs'):
            self.loss_fine_mse = self.losses['mse'](self.out_fine_comp_rgbs, self.data_rgbs) * self.opt.lambda_mse
        else:
            self.loss_fine_mse = 0
        self.loss_tot += self.loss_fine_mse

        # trans depth smoothness loss
        if self.opt.lambda_trans_depth_smoothness > 0 and self.opt.patch_size >= 2:
            self.loss_coarse_trans_depth_smoothness = self.losses['edge-preserving-smoothness'](
                self.out_coarse_trans_depth.view(-1, self.H, self.W),
                self.data_rgbs.view(-1, self.H, self.W, self.opt.dim_rgb)
            ) * self.opt.lambda_trans_depth_smoothness
            self.loss_tot += self.loss_coarse_trans_depth_smoothness
            if hasattr(self, 'out_fine_trans_depth'):
                self.loss_fine_trans_depth_smoothness = self.losses['edge-preserving-smoothness'](
                    self.out_fine_trans_depth.view(-1, self.H, self.W),
                    self.data_rgbs.view(-1, self.H, self.W, self.opt.dim_rgb)
                ) * self.opt.lambda_trans_depth_smoothness
            else:
                self.loss_fine_trans_depth_smoothness = 0
            self.loss_tot += self.loss_fine_trans_depth_smoothness

        # refl depth smoothness loss
        if self.opt.lambda_refl_depth_smoothness > 0 and self.opt.patch_size > 2:
            self.loss_coarse_refl_depth_smoothness = self.losses['edge-preserving-smoothness'](
                self.out_coarse_refl_depth.view(-1, self.H, self.W),
                self.data_rgbs.view(-1, self.H, self.W, self.opt.dim_rgb)
            ) * self.opt.lambda_refl_depth_smoothness
            self.loss_tot += self.loss_coarse_refl_depth_smoothness
            if hasattr(self, 'out_fine_refl_depth'):
                self.loss_fine_refl_depth_smoothness = self.losses['edge-preserving-smoothness'](
                    self.out_fine_refl_depth.view(-1, self.H, self.W),
                    self.data_rgbs.view(-1, self.H, self.W, self.opt.dim_rgb)
                ) * self.opt.lambda_refl_depth_smoothness
            else:
                self.loss_fine_refl_depth_smoothness = 0
            self.loss_tot += self.loss_fine_refl_depth_smoothness

        # beta smoothness loss
        if self.opt.lambda_beta_smoothness > 0 and self.opt.patch_size > 2:
            self.loss_coarse_beta_smoothness = self.losses['smoothness'](
                self.out_coarse_comp_beta.view(-1, self.H, self.W),
            ) * self.opt.lambda_beta_smoothness
            self.loss_tot += self.loss_coarse_beta_smoothness
            if hasattr(self, 'out_fine_comp_beta'):
                self.loss_fine_beta_smoothness = self.losses['smoothness'](
                    self.out_fine_comp_beta.view(-1, self.H, self.W),
                ) * self.opt.lambda_beta_smoothness
            else:
                self.loss_fine_beta_smoothness = 0
            self.loss_tot += self.loss_fine_beta_smoothness 
         
        # beta mask loss
        if self.opt.lambda_beta_mask > 0 and self.opt.with_mask:
            if not hasattr(self, 'data_masks'):
                print('To calculate losses on masks, you must use datasets which provide masks.')
                exit(0)
            self.loss_coarse_beta_mask = self.losses['l1'](self.out_coarse_comp_beta, self.data_masks, mask=self.data_masks_valid) * self.opt.lambda_beta_mask
            self.loss_tot += self.loss_coarse_beta_mask
            if hasattr(self, 'out_fine_comp_beta'):
                self.loss_fine_beta_mask = self.losses['l1'](self.out_fine_comp_beta, self.data_masks, mask=self.data_masks_valid) * self.opt.lambda_beta_mask
            else:
                self.loss_fine_beta_mask = 0
            self.loss_tot += self.loss_fine_beta_mask

        # trans lowpass loss (prevent the trans component from degeneration at the beginning of training)
        if self.opt.lambda_trans_lowpass > 0 and self.opt.patch_size > 2:
            self.loss_coarse_trans_lowpass = self.losses['mse'](
                self.out_coarse_trans_comp_rgbs.view(-1, self.H * self.W, self.opt.dim_rgb).mean(dim=-2),
                self.data_rgbs.view(-1, self.H * self.W, self.opt.dim_rgb).mean(dim=-2)
            ) * self.opt.lambda_trans_lowpass
            self.loss_tot += self.loss_coarse_trans_lowpass
            if hasattr(self, 'out_fine_trans_comp_rgbs'):
                self.loss_fine_trans_lowpass = self.losses['mse'](
                    self.out_fine_trans_comp_rgbs.view(-1, self.H * self.W, self.opt.dim_rgb).mean(dim=-2),
                    self.data_rgbs.view(-1, self.H * self.W, self.opt.dim_rgb).mean(dim=-2)
                ) * self.opt.lambda_trans_lowpass
            else:
                self.loss_fine_trans_lowpass = 0
            self.loss_tot += self.loss_fine_trans_lowpass

        # refl bdc loss
        if self.opt.lambda_refl_bdc > 0:
            self.loss_coarse_refl_bdc = self.losses['l1'](
                self.out_coarse_refl_depth, self.out_coarse_refl_depth_b2f
            ) * self.opt.lambda_refl_bdc
            self.loss_tot += self.loss_coarse_refl_bdc
            if hasattr(self, 'out_fine_refl_depth'):
                self.loss_fine_refl_bdc = self.losses['l1'](
                    self.out_fine_refl_depth, self.out_fine_refl_depth_b2f
                ) * self.opt.lambda_refl_bdc
            else:
                self.loss_fine_refl_bdc = 0
            self.loss_tot += self.loss_fine_refl_bdc
        
        # coarse fine align loss (ensure the coarse and fine network give the same results)
        if self.opt.lambda_coarse_fine_align > 0 and hasattr(self, 'out_fine_comp_rgbs'):
            self.loss_coarse_fine_align = (self.losses['mse'](self.out_coarse_trans_comp_rgbs.detach(), self.out_fine_trans_comp_rgbs) + \
                self.losses['mse'](self.out_coarse_refl_comp_rgbs.detach(), self.out_fine_refl_comp_rgbs) + \
                self.losses['mse'](self.out_coarse_comp_beta.detach(), self.out_fine_comp_beta) + \
                self.losses['mse'](self.out_coarse_trans_depth.detach(), self.out_fine_trans_depth) + \
                self.losses['mse'](self.out_coarse_refl_depth.detach(), self.out_fine_refl_depth)) * self.opt.lambda_coarse_fine_align
            self.loss_tot += self.loss_coarse_fine_align
        
        with torch.no_grad():
            self.loss_coarse_psnr = self.losses['psnr'](self.out_coarse_comp_rgbs, self.data_rgbs)
            if hasattr(self, 'out_fine_comp_rgbs'):
                self.loss_fine_psnr = self.losses['psnr'](self.out_fine_comp_rgbs, self.data_rgbs)
            else:
                self.loss_fine_psnr = 0

    def calculate_vis(self, with_gt):
        W, H = self.opt.img_wh
        coarse_img = self.out_coarse_comp_rgbs.view(H, W, 3).cpu()
        coarse_trans_depth = depth2im(self.out_coarse_trans_depth.view(H, W), opt=self.opt)
        coarse_refl_depth = depth2im(self.out_coarse_refl_depth.view(H, W), opt=self.opt)
        coarse_refl_depth_b2f = depth2im(self.out_coarse_refl_depth_b2f.view(H, W), opt=self.opt)
        coarse_trans_img = self.out_coarse_trans_comp_rgbs.view(H, W, 3).cpu()
        coarse_refl_img = self.out_coarse_refl_comp_rgbs.view(H, W, 3).cpu()
        coarse_beta_img = self.out_coarse_comp_beta.view(H, W, 1).repeat(1, 1, 3).cpu()
        self.coarse_data_dict = {
            'trans_comp_rgbs': coarse_trans_img,
            'trans_depth': self.out_coarse_trans_depth.view(H, W),
            'refl_comp_rgbs': coarse_refl_img,
            'refl_depth': self.out_coarse_refl_depth.view(H, W),
            'comp_rgbs': coarse_img,
            'comp_beta': self.out_coarse_comp_beta.view(H, W),
            'z_vals': self.out_coarse_z_vals.view(H, W, -1)
        }
        if self.opt.N_importance > 0:
            fine_img = self.out_fine_comp_rgbs.view(H, W, 3).cpu()
            fine_trans_depth = depth2im(self.out_fine_trans_depth.view(H, W), opt=self.opt)
            fine_refl_depth = depth2im(self.out_fine_refl_depth.view(H, W), opt=self.opt)
            fine_refl_depth_b2f = depth2im(self.out_fine_refl_depth_b2f.view(H, W), opt=self.opt)
            fine_trans_img = self.out_fine_trans_comp_rgbs.view(H, W, 3).cpu()
            fine_refl_img = self.out_fine_refl_comp_rgbs.view(H, W, 3).cpu()
            fine_beta_img = self.out_fine_comp_beta.view(H, W, 1).repeat(1, 1, 3).cpu()
            self.fine_data_dict = {
                'trans_comp_rgbs': fine_trans_img,
                'trans_depth': self.out_fine_trans_depth.view(H, W),
                'refl_comp_rgbs': fine_refl_img,
                'refl_depth': self.out_fine_refl_depth.view(H, W),
                'comp_rgbs': fine_img,
                'comp_beta': self.out_fine_comp_beta.view(H, W)
            }
        if with_gt:
            gt_img = self.data_rgbs.view(H, W, 3).cpu()
            self.coarse_data_dict['gt_rgbs'] = gt_img
            coarse_pred = torch.cat([gt_img, coarse_img, coarse_trans_img, coarse_trans_depth, coarse_refl_img, coarse_refl_depth, coarse_refl_depth_b2f, coarse_beta_img], dim=1)
            self.coarse_pred_img = coarse_pred
            if self.opt.N_importance > 0:
                self.fine_data_dict['gt_rgbs'] = gt_img
                fine_pred = torch.cat([gt_img, fine_img, fine_trans_img, fine_trans_depth, fine_refl_img, fine_refl_depth, fine_refl_depth_b2f, fine_beta_img], dim=1)
                self.fine_pred_img = fine_pred
        else:
            coarse_pred = torch.cat([coarse_img, coarse_trans_img, coarse_trans_depth, coarse_refl_img, coarse_refl_depth, coarse_refl_depth_b2f, coarse_beta_img], dim=1)
            self.coarse_pred_img = coarse_pred
            if self.opt.N_importance > 0:
                fine_pred = torch.cat([fine_img, fine_trans_img, fine_trans_depth, fine_refl_img, fine_refl_depth, fine_refl_depth_b2f, fine_beta_img], dim=1)
                self.fine_pred_img = fine_pred

