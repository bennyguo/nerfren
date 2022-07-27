import math
from options import Configurable, str2bool
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from .embedding import BaseEmbedding
from utils.utils import find_class_using_name


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    
    if init_type != 'default':
        net.apply(init_func)  # apply the initialization function <init_func>
    
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    if hasattr(net, 'init_weights'):
        net.init_weights()


def init_net(net, opt):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    """
    if opt.n_gpus > 0:
        assert(torch.cuda.is_available())
        net.to(opt.device)
        net = torch.nn.DataParallel(net)  # multi-GPUs
    init_weights(net, opt.init_type, init_gain=opt.init_gain)
    return net


def get_scheduler(optimizer, opt, last_epoch=-1):
    """Return a learning rate scheduler
    """
    if opt.lr_policy == 'linear':
        # keep the initial lr for the first <opt.n_epochs - opt.n_epochs_decay> epochs
        # then linearly decay to lr_final over the next <opt.n_epochs_decay> epochs
        def lambda_rule(epoch):
            t = max(0, epoch + 1 - opt.n_epochs + opt.n_epochs_decay) / float(opt.n_epochs_decay + 1)
            lr = opt.lr * (1 - t) + opt.lr_final * t
            return lr / opt.lr
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)
    elif opt.lr_policy == 'exp':
        # keep the initial lr for the first <opt.n_epochs - opt.n_epochs_decay> epochs
        # then exponentially decay to lr_final over the next <opt.n_epochs_decay> epochs        
        def lambda_rule(epoch):
            t = max(0, epoch + 1 - opt.n_epochs + opt.n_epochs_decay) / float(opt.n_epochs_decay + 1)
            lr = math.exp(math.log(opt.lr) * (1 - t) + math.log(opt.lr_final) * t)
            return lr / opt.lr
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_epochs, gamma=opt.lr_decay_gamma, last_epoch=last_epoch)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler    


class VanillaMLP(nn.Module, Configurable):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--D', type=int, default=8)
        parser.add_argument('--W', type=int, default=256)
        parser.add_argument('--skips', type=int, nargs='+', default=[4])
        parser.add_argument('--stop_grad', type=str2bool, default=False)
        parser.add_argument('--no_dir', action='store_true')
        return parser

    def __init__(self, opt):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz
        in_channels_dir: number of input channels for direction
        skips: add skip connection in the Dth layer
        """
        super(VanillaMLP, self).__init__()
        self.opt = opt
        self.D = opt.D
        self.W = opt.W
        self.in_channels_xyz = find_class_using_name('models.embedding', opt.embedding, type=BaseEmbedding)(opt.dim_pos, opt.deg_pos).out_channels
        self.in_channels_dir = find_class_using_name('models.embedding', opt.embedding, type=BaseEmbedding)(opt.dim_dir, opt.deg_dir, no_xyz=True).out_channels
        self.skips = opt.skips
        self.out_channels_rgb = opt.dim_rgb

        # xyz encoding layers
        for i in range(self.D):
            if i == 0:
                layer = nn.Linear(self.in_channels_xyz, self.W)
            elif i in self.skips:
                layer = nn.Linear(self.W + self.in_channels_xyz, self.W)
            else:
                layer = nn.Linear(self.W, self.W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(self.W, self.W)

        # direction encoding layers
        if not opt.no_dir:
            self.dir_encoding = nn.Sequential(
                                    nn.Linear(self.W + self.in_channels_dir, self.W//2),
                                    nn.ReLU(True))
        else:
            self.dir_encoding = nn.Sequential(
                nn.Linear(self.W, self.W//2),
                nn.ReLU(True)
            )

        # output layers
        self.sigma = nn.Linear(self.W, 1)
        if opt.color_activation == 'sigmoid':
            color_activation = nn.Sigmoid()
        elif opt.color_activation == 'relu':
            color_activation = nn.ReLU(True)
        elif opt.color_activation == 'none':
            color_activation = nn.Identity()
        self.rgb = nn.Sequential(
            nn.Linear(self.W // 2, self.out_channels_rgb),
            color_activation
        )

    def forward(self, x):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        input_xyz, input_dir = torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        if not self.opt.no_dir:
            dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        else:
            dir_encoding_input = xyz_encoding_final

        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)

        return out


class TwoLayerMLP(nn.Module, Configurable):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--D', type=int, default=5)
        parser.add_argument('--D_sep', type=int, default=1)
        parser.add_argument('--W', type=int, default=256)
        parser.add_argument('--skips', type=int, nargs='+', default=[4])
        parser.add_argument('--trans_with_dir', type=str2bool, default=True)
        parser.add_argument('--refl_with_dir', type=str2bool, default=False)
        parser.add_argument('--beta_with_dir', type=str2bool, default=False)
        return parser

    def __init__(self, opt):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz
        in_channels_dir: number of input channels for direction
        skips: add skip connection in the Dth layer
        """
        super(TwoLayerMLP, self).__init__()
        self.opt = opt
        self.D = opt.D
        self.D_sep = opt.D_sep
        self.W = opt.W
        self.in_channels_xyz = find_class_using_name('models.embedding', opt.embedding, type=BaseEmbedding)(opt.dim_pos, opt.deg_pos).out_channels
        self.in_channels_dir = find_class_using_name('models.embedding', opt.embedding, type=BaseEmbedding)(opt.dim_dir, opt.deg_dir, no_xyz=True).out_channels
        self.skips = opt.skips
        self.out_channels_trans_rgb, self.out_channels_refl_rgb = opt.dim_rgb, opt.dim_rgb

        # xyz encoding layers
        for i in range(self.D):
            if i == 0:
                layer = nn.Linear(self.in_channels_xyz, self.W)
            elif i in self.skips:
                layer = nn.Linear(self.W + self.in_channels_xyz, self.W)
            else:
                layer = nn.Linear(self.W, self.W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)

        if opt.color_activation == 'sigmoid':
            color_activation = nn.Sigmoid()
        elif opt.color_activation == 'relu':
            color_activation = nn.ReLU(True)
        elif opt.color_activation == 'none':
            color_activation = nn.Identity()
        
        if opt.beta_activation == 'sigmoid':
            beta_activation = nn.Sigmoid()
        elif opt.beta_activation == 'none':
            beta_activation = nn.Identity()

        trans_encoding_sep, refl_encoding_sep = [], []
        for i in range(self.D_sep):
            trans_encoding_sep += [nn.Linear(self.W, self.W), nn.ReLU(True)]
            refl_encoding_sep += [nn.Linear(self.W, self.W), nn.ReLU(True)]
            
        self.trans_encoding = nn.Sequential(
            *trans_encoding_sep
        )
        self.trans_sigma = nn.Linear(self.W, 1)
        self.beta = nn.Sequential(
            nn.Linear(self.W + self.in_channels_dir, 1) if opt.beta_with_dir else nn.Linear(self.W, 1),
            beta_activation
        )
        self.trans_rgb = nn.Sequential(
            nn.Linear(self.W + self.in_channels_dir, self.W) if opt.trans_with_dir else nn.Linear(self.W, self.W),
            nn.ReLU(True),
            nn.Linear(self.W, self.W // 2),
            nn.ReLU(True),
            nn.Linear(self.W // 2, self.out_channels_trans_rgb),
            color_activation
        )

        self.refl_encoding = nn.Sequential(
            *refl_encoding_sep
        )
        self.refl_sigma = nn.Linear(self.W, 1)
        self.refl_rgb = nn.Sequential(
            nn.Linear(self.W + self.in_channels_dir, self.W) if opt.refl_with_dir else nn.Linear(self.W, self.W),
            nn.ReLU(True),
            nn.Linear(self.W, self.W // 2),
            nn.ReLU(True),
            nn.Linear(self.W // 2, self.out_channels_refl_rgb),
            color_activation
        )

    def forward(self, x, return_type=None):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        input_xyz, input_dir = torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        if return_type is None or return_type == 'trans':
            trans_encoding = self.trans_encoding(xyz_)
            trans_sigma = self.trans_sigma(trans_encoding)
            beta = self.beta(trans_encoding if not self.opt.beta_with_dir else torch.cat([trans_encoding, input_dir], dim=-1))
            trans_rgb = self.trans_rgb(trans_encoding if not self.opt.trans_with_dir else torch.cat([trans_encoding, input_dir], dim=-1))
        
        if return_type is None or return_type == 'refl':
            refl_encoding = self.refl_encoding(xyz_)
            refl_sigma = self.refl_sigma(refl_encoding)
            refl_rgb = self.refl_rgb(refl_encoding if not self.opt.refl_with_dir else torch.cat([refl_encoding, input_dir], dim=-1))

        if return_type == 'trans':
            return {
                'beta': beta,
                'trans_sigma': trans_sigma,
                'trans_rgb': trans_rgb
            }
        if return_type == 'refl':
            return {
                'refl_sigma': refl_sigma,
                'refl_rgb': refl_rgb
            }
        return {
            'beta': beta,
            'trans_sigma': trans_sigma,
            'trans_rgb': trans_rgb,
            'refl_sigma': refl_sigma,
            'refl_rgb': refl_rgb
        }
