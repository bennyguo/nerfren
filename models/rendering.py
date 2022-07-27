import torch
import torch.nn as nn
import torch.nn.functional as TF


class VolumetricRenderer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        if opt.sigma_activation == 'relu':
            self.sigma_activation = nn.ReLU(inplace=False)
        elif opt.sigma_activation == 'softplus':
            self.sigma_activation = lambda x: torch.log(1 + torch.exp(x - 1))
    
    def forward(self, rgb, sigma, z_vals, white_bkgd):
        """Volumetric Rendering Function.
        Args:
            rgb: color, [N_ray, N_samples, 3].
            sigma: density, [N_ray, N_samples].
            z_vals: depth, [N_ray, N_samples].
            white_bkgd: white background, bool.
        Returns:
            comp_rgb: jnp.ndarray(float32), [N_ray, 3].
            depth: jnp.ndarray(float32), [N_ray].
            opacity: jnp.ndarray(float32), [N_ray].
            weights: jnp.ndarray(float32), [N_ray, N_samples].
        """
        eps = 1e-10
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples)
        deltas = torch.cat([deltas, 1e10 * torch.ones_like(deltas[:, :1])], -1)

        alpha = 1 - torch.exp(-deltas * self.sigma_activation(sigma)) # (N_rays, N_samples)
        accum_prod = torch.cat([
            torch.ones_like(alpha[:, :1]),
            torch.cumprod(1 - alpha[:, :-1] + eps, dim=-1)
        ], dim=-1)
        weights = alpha * accum_prod # (N_rays, N_samples)
        comp_rgb = (weights[..., None] * rgb).sum(dim=-2) # (N_rays, 3)
        depth = (weights * z_vals).sum(dim=-1) # (N_rays)
        opacity = weights.sum(dim=-1) # (N_rays)

        if white_bkgd:
            comp_rgb += 1 - opacity[..., None]
        
        return comp_rgb, depth, opacity, weights
    
    def forward_bidir(self, trans_rgb, trans_sigma, refl_rgb, refl_sigma, beta, z_vals, white_bkgd):
        """
        Perform volume rendering in both directions to calculate the BDC loss
        """
        eps = 1e-10
        if isinstance(z_vals, tuple):
            trans_z_vals, refl_z_vals = z_vals

            trans_deltas = trans_z_vals[:, 1:] - trans_z_vals[:, :-1] # (N_rays, N_samples)
            trans_deltas = torch.cat([trans_deltas, 1e10 * torch.ones_like(trans_deltas[:, :1])], -1)
            trans_deltas_b2f = trans_z_vals[:, 1:] - trans_z_vals[:, :-1]
            trans_deltas_b2f = torch.cat([1e10 * torch.ones_like(trans_deltas_b2f[:, :1]), trans_deltas_b2f], -1).flip(-1)

            refl_deltas = refl_z_vals[:, 1:] - refl_z_vals[:, :-1] # (N_rays, N_samples)
            refl_deltas = torch.cat([refl_deltas, 1e10 * torch.ones_like(refl_deltas[:, :1])], -1)
            refl_deltas_b2f = refl_z_vals[:, 1:] - refl_z_vals[:, :-1]
            refl_deltas_b2f = torch.cat([1e10 * torch.ones_like(refl_deltas_b2f[:, :1]), refl_deltas_b2f], -1).flip(-1)
        else:
            trans_z_vals, refl_z_vals = z_vals, z_vals

            deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples)
            deltas = torch.cat([deltas, 1e10 * torch.ones_like(deltas[:, :1])], -1)
            deltas_b2f = z_vals[:, 1:] - z_vals[:, :-1]
            deltas_b2f = torch.cat([1e10 * torch.ones_like(deltas_b2f[:, :1]), deltas_b2f], -1).flip(-1)

            trans_deltas, refl_deltas = deltas, deltas
            trans_deltas_b2f, refl_deltas_b2f = deltas_b2f, deltas_b2f

        trans_sigma, refl_sigma = self.sigma_activation(trans_sigma), self.sigma_activation(refl_sigma)

        trans_alpha = 1 - torch.exp(-trans_deltas * trans_sigma) # (N_rays, N_samples)
        trans_accum_prod = torch.cat([
            torch.ones_like(trans_alpha[:, :1]),
            torch.cumprod(1 - trans_alpha[:, :-1] + eps, dim=-1)
        ], dim=-1)
        trans_weights = trans_alpha * trans_accum_prod # (N_rays, N_samples)
        trans_comp_rgb = (trans_weights[..., None] * trans_rgb).sum(dim=-2) # (N_rays, 3)
        trans_depth = (trans_weights * trans_z_vals).sum(dim=-1) # (N_rays)
        trans_opacity = trans_weights.sum(dim=-1) # (N_rays)

        trans_alpha_b2f = 1 - torch.exp(-trans_deltas_b2f * trans_sigma.flip(-1))
        trans_accum_prod_b2f = torch.cat([
            torch.ones_like(trans_alpha_b2f[:, :1]),
            torch.cumprod(1 - trans_alpha_b2f[:, :-1] + eps, dim=-1)
        ], dim=-1)
        trans_weights_b2f = trans_alpha_b2f * trans_accum_prod_b2f
        trans_depth_b2f = (trans_weights_b2f * trans_z_vals.flip(-1)).sum(dim=-1)

        comp_beta = (trans_weights[..., None] * beta).sum(dim=-2) # (N_rays, 1)

        refl_alpha = 1 - torch.exp(-refl_deltas * refl_sigma) # (N_rays, N_samples)
        refl_accum_prod = torch.cat([
            torch.ones_like(refl_alpha[:, :1]),
            torch.cumprod(1 - refl_alpha[:, :-1] + eps, dim=-1)
        ], dim=-1)
        refl_weights = refl_alpha * refl_accum_prod # (N_rays, N_samples)
        refl_comp_rgb = (refl_weights[..., None] * refl_rgb).sum(dim=-2) # (N_rays, 3)
        refl_depth = (refl_weights * refl_z_vals).sum(dim=-1) # (N_rays)
        refl_opacity = refl_weights.sum(dim=-1) # (N_rays)        

        refl_alpha_b2f = 1 - torch.exp(-refl_deltas_b2f * refl_sigma.flip(-1))
        refl_accum_prod_b2f = torch.cat([
            torch.ones_like(refl_alpha_b2f[:, :1]),
            torch.cumprod(1 - refl_alpha_b2f[:, :-1] + eps, dim=-1)
        ], dim=-1)
        refl_weights_b2f = refl_alpha_b2f * refl_accum_prod_b2f
        refl_depth_b2f = (refl_weights_b2f * refl_z_vals.flip(-1)).sum(dim=-1)

        comp_rgb = trans_comp_rgb + comp_beta * refl_comp_rgb

        if white_bkgd:
            comp_rgb += 1 - trans_opacity[..., None]
        
        comp_rgb = torch.clamp(comp_rgb, 0, 1)
        
        return trans_comp_rgb, refl_comp_rgb, comp_rgb, trans_depth, refl_depth, trans_depth_b2f, refl_depth_b2f, comp_beta, trans_sigma, refl_sigma, trans_opacity, refl_opacity, trans_weights, refl_weights

        