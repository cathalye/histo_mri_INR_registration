import math
import random
import numpy as np
import torch


def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_affine_parameters(device, init_rotation=0.0):
    """
    Create learnable affine parameters (A, b) for 3D transform.
    A: 3x3 linear part, b: 3D translation.
    init_rotation: degrees, rotation in XY plane (Z unchanged).
    """
    if abs(init_rotation) > 1e-6:
        angle_rad = math.radians(init_rotation)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        A_init = torch.eye(3, dtype=torch.float32).to(device)
        A_init[0, 0] = cos_a
        A_init[0, 1] = -sin_a
        A_init[1, 0] = sin_a
        A_init[1, 1] = cos_a
    else:
        A_init = torch.eye(3, dtype=torch.float32).to(device)
    b_init = torch.zeros(3, dtype=torch.float32).to(device)

    return torch.nn.Parameter(A_init), torch.nn.Parameter(b_init)


def _build_optimizer_param_groups(support_network, residual_network, A, b, base_lr, affine_lr_multiplier):
    """
    Build optimizer param groups; includes affine params when A, b are not None.
    """
    network_params = list(support_network.parameters()) + list(residual_network.parameters())
    if A is not None and b is not None:
        affine_lr = base_lr * affine_lr_multiplier
        return [
            {'params': [A, b], 'lr': affine_lr},
            {'params': network_params, 'lr': base_lr},
        ]

    return [{'params': network_params, 'lr': base_lr}]


def prepare_networks(network_size, n_channels, device, affine_transform=False, affine_lr_multiplier=10.0, affine_init_rotation=0.0):
    """
    Prepare support, residual, and deformation networks for training.
    """
    # Note that the output dimension of the support and residual networks is number of channels not the dimensions of the image
    support_network = ImpNet(
        input_dim=2, output_dim=n_channels, hidden_dim=256, hidden_n_layers=network_size[1], last_layer_weights_small=False,
        is_last_linear=True, input_omega_0=30.0, hidden_omega_0=30., input_n_encoding_functions=6
    ).to(device)

    residual_network = ImpNet(
        input_dim=2, output_dim=n_channels, hidden_dim=256, hidden_n_layers=network_size[1], last_layer_weights_small=False,
        is_last_linear=True, input_omega_0=30.0, hidden_omega_0=30., input_n_encoding_functions=6
    ).to(device)

    # 3D deformation network: takes 2D coordinates, outputs 3D deformations
    # Use lower omega_0 (5.0 vs 30.0) and no positional encoding for smoother output
    deformation_network = ImpNet(
        input_dim=2, output_dim=3, hidden_dim=256, hidden_n_layers=network_size[0], last_layer_weights_small=True,
        is_last_linear=True, input_omega_0=5.0, hidden_omega_0=5.0,
        input_n_encoding_functions=0
    ).to(device)

    base_lr = 0.0001
    if affine_transform:
        A, b = create_affine_parameters(device, affine_init_rotation)
        params = _build_optimizer_param_groups(
            support_network, residual_network, A, b, base_lr, affine_lr_multiplier
        )
    else:
        A, b = None, None
        params = _build_optimizer_param_groups(
            support_network, residual_network, None, None, base_lr, affine_lr_multiplier
        )

    optimizer = torch.optim.AdamW(params, lr=base_lr)

    return deformation_network, support_network, residual_network, optimizer, A, b


def create_phase2_optimizer(deformation_network, support_network, residual_network, base_lr=0.0001, deformation_lr_multiplier=1.0):
    """
    Create optimizer for Phase 2 training (decomposition + deformation, affine frozen).
    """
    deformation_lr = base_lr * deformation_lr_multiplier

    deformation_params = [{'params': deformation_network.parameters(), 'lr': deformation_lr}]
    decomposition_params = [{'params': list(support_network.parameters()) + list(residual_network.parameters()), 'lr': base_lr}]

    params = deformation_params + decomposition_params
    optimizer = torch.optim.AdamW(params, lr=base_lr)

    return optimizer


def freeze_affine_parameters(A, b):
    """
    Freeze affine parameters by setting requires_grad to False.
    """
    if A is not None:
        A.requires_grad = False
    if b is not None:
        b.requires_grad = False
    print("Affine parameters frozen.")


class SineLayer(torch.nn.Module):
    """
    Based on Sitzmann et al. 2020
    "Implicit Neural Representations with Periodic Activation Functions"
    https://arxiv.org/abs/2006.09661
    https://github.com/vsitzmann/siren
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 bias=True,
                 is_first_layer=False,
                 omega_0=30.0):

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.omega_0 = omega_0
        self.is_first_layer = is_first_layer

        self.linear = torch.nn.Linear(input_dim, output_dim, bias=bias)

        self.init_weights()

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

    def init_weights(self):
        with torch.no_grad():
            if self.is_first_layer:
                # initialize the weights to be small so that sin function is
                # almost linear at the beginning
                self.linear.weight.uniform_(-1 / self.input_dim,
                                             1 / self.input_dim)
            else:
                # scaled version of the Glorot initialization
                # standard practice
                self.linear.weight.uniform_(-np.sqrt(6 / self.input_dim) / self.omega_0,
                                             np.sqrt(6 / self.input_dim) / self.omega_0)


class InputEncoding(torch.nn.Module):
    """
    Harmonic encoding of the spatial coordinates
    n_functions: number of frequency bands to use
    """
    def __init__(self,
                 n_functions=6,
                 base_omega_0=1.0,
                 append_coords=True):

        super().__init__()

        # n_functions powers of 2 for harmonic encoding
        frequencies = 2.0 ** torch.arange(n_functions, dtype=torch.float32)

        # register_buffer is a built-in method saved and loaded with the model's state_dict
        # creates a register buffer called frequencies with the scaled frequencies
        self.register_buffer('frequencies', frequencies*base_omega_0, persistent=True)
        self.append_coords = append_coords

    def forward(self, x):
        """
        Arguments:
            x: tensor of shape [batch, dim] with values in [-1, 1]
        Returns:
            x_encoded: a harmonic embedding of x with shape [batch, (n_functions * 2 + int(append_coords)) * dim]
        """
        x = (x + 1) / 2 + 1 # from [-1, 1] to [1, 2]
        # each coordinate x is multiplied by each of the frequencies
        x_encoded = (x[..., None] * self.frequencies).reshape(*x.shape[:-1], -1)

        if self.append_coords:
            x_encoded = torch.cat((x_encoded.sin(), x_encoded.cos(), x), dim=-1)
        else:
            x_encoded = torch.cat((x_encoded.sin(), x_encoded.cos()), dim=-1)

        return x_encoded


class ImpNet(torch.nn.Module):
    """
    Siren network.
    """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 hidden_n_layers,
                 output_dim,
                 is_last_linear=False,
                 mid_skip=True,
                 input_omega_0=30.0,
                 hidden_omega_0=30.0,
                 input_n_encoding_functions=6,
                 last_layer_weights_small=False):
                 #output_thresh=False):

        super().__init__()

        self.mid_skip = mid_skip
        self.n_skip = -1
        # self.output_thresh = output_thresh
        self.input_encoding = input_n_encoding_functions>0
        self.layers = torch.nn.ModuleList()

        if self.input_encoding==True:
            self.encoding = InputEncoding(n_functions=input_n_encoding_functions, append_coords=True)
            self.layers.append(SineLayer(2*input_n_encoding_functions*input_dim+input_dim, hidden_dim, is_first_layer=True, omega_0=input_omega_0))
            self.skip_dim = 2*input_n_encoding_functions*input_dim+input_dim
        else:
            self.layers.append(SineLayer(input_dim, hidden_dim, is_first_layer=True, omega_0=input_omega_0))
            self.skip_dim = input_dim


        for i in range(hidden_n_layers):
            # for the middle layer, add the skip connection
            if i+1 == np.ceil(hidden_n_layers/2) and self.mid_skip:
                self.layers.append(SineLayer(hidden_dim+self.skip_dim, hidden_dim, is_first_layer=False, omega_0=hidden_omega_0))
                self.n_skip = len(self.layers)
            else:
                self.layers.append(SineLayer(hidden_dim, hidden_dim, is_first_layer=False, omega_0=hidden_omega_0))


        if is_last_linear==True:
            last_linear = torch.nn.Linear(hidden_dim, output_dim)

            with torch.no_grad():
                if last_layer_weights_small:
                    last_linear.weight.uniform_(-0.0001, 0.0001) # weights are set to small values to provide small deformations at the first epochs
                else:
                    last_linear.weight.uniform_(-np.sqrt(6 / hidden_dim) / hidden_omega_0,
                                                 np.sqrt(6 / hidden_dim) / hidden_omega_0)

                self.layers.append(last_linear)
        else:
            self.layers.append(SineLayer(hidden_dim, output_dim, is_first_layer=False, omega_0=hidden_omega_0))


    def forward(self, coords, clone=True):
        if clone==True:
            # Clone the coordinates and detach them from the computational graph
            coords = coords.clone().detach().requires_grad_(True)
            x = coords

            if self.input_encoding==True:
                x = self.encoding(x)
            if self.mid_skip==True:
                add_to_skip = x

            for i, layer in enumerate(self.layers):
                if (i+1)==self.n_skip and self.mid_skip==True:
                    x = torch.cat([x, add_to_skip], dim=-1)
                x = layer(x)

            return x, coords

        else:
            x = coords

            if self.input_encoding==True:
                x = self.encoding(x)
            if self.mid_skip==True:
                add_to_skip = x

            for i, layer in enumerate(self.layers):
                if (i+1)==self.n_skip and self.mid_skip==True:
                    x = torch.cat([x, add_to_skip], dim=-1)
                x = layer(x)

            return x
