import torch
import torch.nn as nn
import torch.nn.functional as TF


class L1Regularization(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
    
    def forward(self, x):
        return torch.mean(torch.abs(x))


class L1Loss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, inputs, targets, mask=None):
        if mask is None:
            loss = self.loss(inputs, targets)
        else:
            loss = self.loss(inputs * mask, targets * mask)
        return loss
        

class MSELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets, mask=None):
        if mask is None:
            loss = self.loss(inputs, targets)
        else:
            loss = self.loss(inputs * mask, targets * mask)
        return loss


class PSNR(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
    
    def forward(self, inputs, targets, valid_mask=None):
        value = (inputs - targets)**2
        if valid_mask is not None:
            value = value[valid_mask]
        return -10 * torch.log10(torch.mean(value))


class SmoothnessLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.patch_size = opt.patch_size
        self.loss = lambda x: torch.mean(torch.abs(x))
    
    def forward(self, inputs):
        L1 = self.loss(inputs[:,:,:-1] - inputs[:,:,1:])
        L2 = self.loss(inputs[:,:-1,:] - inputs[:,1:,:])
        L3 = self.loss(inputs[:,:-1,:-1] - inputs[:,1:,1:])
        L4 = self.loss(inputs[:,1:,:-1] - inputs[:,:-1,1:])
        return (L1 + L2 + L3 + L4) / 4               


class EdgePreservingSmoothnessLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.patch_size = opt.patch_size
        self.gamma = opt.bilateral_gamma
        self.loss = lambda x: torch.mean(torch.abs(x))
        self.bilateral_filter = lambda x: torch.exp(-torch.abs(x).sum(-1) / self.gamma)
    
    def forward(self, inputs, weights):
        w1 = self.bilateral_filter(weights[:,:,:-1] - weights[:,:,1:])
        w2 = self.bilateral_filter(weights[:,:-1,:] - weights[:,1:,:])
        w3 = self.bilateral_filter(weights[:,:-1,:-1] - weights[:,1:,1:])
        w4 = self.bilateral_filter(weights[:,1:,:-1] - weights[:,:-1,1:])

        L1 = self.loss(w1 * (inputs[:,:,:-1] - inputs[:,:,1:]))
        L2 = self.loss(w2 * (inputs[:,:-1,:] - inputs[:,1:,:]))
        L3 = self.loss(w3 * (inputs[:,:-1,:-1] - inputs[:,1:,1:]))
        L4 = self.loss(w4 * (inputs[:,1:,:-1] - inputs[:,:-1,1:]))
        return (L1 + L2 + L3 + L4) / 4        


class SSIM():
    def __init__(self, data_range=(0, 1), kernel_size=(11, 11), sigma=(1.5, 1.5), k1=0.01, k2=0.03, gaussian=True):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian = gaussian
        
        if any(x % 2 == 0 or x <= 0 for x in self.kernel_size):
            raise ValueError(f"Expected kernel_size to have odd positive number. Got {kernel_size}.")
        if any(y <= 0 for y in self.sigma):
            raise ValueError(f"Expected sigma to have positive number. Got {sigma}.")
        
        data_scale = data_range[1] - data_range[0]
        self.c1 = (k1 * data_scale)**2
        self.c2 = (k2 * data_scale)**2
        self.pad_h = (self.kernel_size[0] - 1) // 2
        self.pad_w = (self.kernel_size[1] - 1) // 2
        self._kernel = self._gaussian_or_uniform_kernel(kernel_size=self.kernel_size, sigma=self.sigma)
    
    def _uniform(self, kernel_size):
        max, min = 2.5, -2.5
        ksize_half = (kernel_size - 1) * 0.5
        kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        for i, j in enumerate(kernel):
            if min <= j <= max:
                kernel[i] = 1 / (max - min)
            else:
                kernel[i] = 0

        return kernel.unsqueeze(dim=0)  # (1, kernel_size)

    def _gaussian(self, kernel_size, sigma):
        ksize_half = (kernel_size - 1) * 0.5
        kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        gauss = torch.exp(-0.5 * (kernel / sigma).pow(2))
        return (gauss / gauss.sum()).unsqueeze(dim=0)  # (1, kernel_size)

    def _gaussian_or_uniform_kernel(self, kernel_size, sigma):
        if self.gaussian:
            kernel_x = self._gaussian(kernel_size[0], sigma[0])
            kernel_y = self._gaussian(kernel_size[1], sigma[1])
        else:
            kernel_x = self._uniform(kernel_size[0])
            kernel_y = self._uniform(kernel_size[1])

        return torch.matmul(kernel_x.t(), kernel_y)  # (kernel_size, 1) * (1, kernel_size)

    def __call__(self, output, target, reduction='mean'):
        if output.dtype != target.dtype:
            raise TypeError(
                f"Expected output and target to have the same data type. Got output: {output.dtype} and y: {target.dtype}."
            )

        if output.shape != target.shape:
            raise ValueError(
                f"Expected output and target to have the same shape. Got output: {output.shape} and y: {target.shape}."
            )

        if len(output.shape) != 4 or len(target.shape) != 4:
            raise ValueError(
                f"Expected output and target to have BxCxHxW shape. Got output: {output.shape} and y: {target.shape}."
            )

        assert reduction in ['mean', 'sum', 'none']

        channel = output.size(1)
        if len(self._kernel.shape) < 4:
            self._kernel = self._kernel.expand(channel, 1, -1, -1)

        output = TF.pad(output, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect")
        target = TF.pad(target, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect")

        input_list = torch.cat([output, target, output * output, target * target, output * target])
        outputs = TF.conv2d(input_list, self._kernel, groups=channel)

        output_list = [outputs[x * output.size(0) : (x + 1) * output.size(0)] for x in range(len(outputs))]

        mu_pred_sq = output_list[0].pow(2)
        mu_target_sq = output_list[1].pow(2)
        mu_pred_target = output_list[0] * output_list[1]

        sigma_pred_sq = output_list[2] - mu_pred_sq
        sigma_target_sq = output_list[3] - mu_target_sq
        sigma_pred_target = output_list[4] - mu_pred_target

        a1 = 2 * mu_pred_target + self.c1
        a2 = 2 * sigma_pred_target + self.c2
        b1 = mu_pred_sq + mu_target_sq + self.c1
        b2 = sigma_pred_sq + sigma_target_sq + self.c2

        ssim_idx = (a1 * a2) / (b1 * b2)
        _ssim = torch.mean(ssim_idx, (1, 2, 3))

        if reduction == 'none':
            return _ssim
        return _ssim.mean() if reduction == 'mean' else _ssim.sum()
