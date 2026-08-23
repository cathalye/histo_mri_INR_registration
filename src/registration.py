"""
Registration training step logic, loss computation, and checkpoint I/O.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from losses import get_alpha, RegistrationLosses
from transforms import apply_affine_transform_pytorch_2d3d, sample_volume_with_grid


@dataclass
class RegistrationConfig:
    """Scalar/config values for registration training."""

    img_size: int
    n_channels: int
    phase1_epochs: int
    total_epochs: int
    flow_scale: float
    device: str
    start_alpha: List[float]
    end_alpha: List[float]


class RegistrationTrainer:
    """
    Holds shared config, networks, images, and losses for registration training.
    Exposes step() and compute_losses() with minimal per-call arguments.
    """

    def __init__(self, config: RegistrationConfig, losses: RegistrationLosses,
                 fixed_image: torch.Tensor, moving_image: torch.Tensor, mask: Optional[torch.Tensor],
                 coords: torch.Tensor, coords_init: torch.Tensor,
                 support_network: nn.Module, residual_network: nn.Module, deformation_network: nn.Module,
                 A: Optional[torch.Tensor], b: Optional[torch.Tensor]):
        self.config = config
        self.losses = losses
        self.img_size = config.img_size
        self.n_channels = config.n_channels
        self.phase1_epochs = config.phase1_epochs
        self.total_epochs = config.total_epochs
        self.flow_scale = config.flow_scale
        self.device = config.device
        self.start_alpha = config.start_alpha
        self.end_alpha = config.end_alpha
        self.fixed_image = fixed_image
        self.moving_image = moving_image
        self.mask = mask
        self.coords = coords
        self.coords_init = coords_init
        self.support_network = support_network
        self.residual_network = residual_network
        self.deformation_network = deformation_network
        self.A = A
        self.b = b


    def step(self, epoch, viz_images):
        """
        Run one forward pass of the registration (phase 1 or phase 2).

        Args:
            epoch: The current epoch number
            viz_images: A dictionary to store the images for visualization

        Returns:
            viz_images: Updated dict with support, residual, reconstructed, moved_affine, moved_deform, flow, etc.
            sampling_grid_flat: [H*W, 3] sampling grid (flattened)
            coords_deform: Coords for deformation network, or None in phase 1
        """
        img_size = self.img_size
        n_channels = self.n_channels
        fixed_image = self.fixed_image
        moving_image = self.moving_image
        coords = self.coords
        coords_init = self.coords_init
        A = self.A
        b = self.b
        flow_scale = self.flow_scale

        # Get support and residual pixels from support and residual networks
        support_pixels = self.support_network(coords, clone=False)
        support_image = support_pixels.view(img_size, img_size, n_channels).permute(2, 0, 1).unsqueeze(0)
        viz_images["support"] = support_image
        residual_pixels = self.residual_network(coords, clone=False)
        residual_image = residual_pixels.view(img_size, img_size, n_channels).permute(2, 0, 1).unsqueeze(0)
        viz_images["residual"] = residual_image

        # Get reconstructed moving image
        recon_pixels = support_pixels + residual_pixels
        recon_image = recon_pixels.view(img_size, img_size, n_channels).permute(2, 0, 1).unsqueeze(0)
        viz_images["reconstructed"] = recon_image

        if epoch < self.phase1_epochs:
            # Phase 1: Affine transform only (deformation network not used)
            if A is not None and b is not None:
                sampling_grid, moved_image_affine = apply_affine_transform_pytorch_2d3d(fixed_image, moving_image, A, b, padding_mode="zeros")
                viz_images["moved_affine"] = moved_image_affine
                viz_images["moved_deform"] = moved_image_affine
            else:
                viz_images["moved_affine"] = None
                viz_images["moved_deform"] = None
            viz_images["flow"] = None
            coords_deform = None
            sampling_grid_flat = sampling_grid.view(-1, 3) if sampling_grid is not None else None
        else:
            # Phase 2: Affine frozen, deformation network active
            sampling_grid, moved_image_affine = apply_affine_transform_pytorch_2d3d(fixed_image, moving_image, A, b, padding_mode="zeros")

            coords_deform = coords_init.clone().detach().requires_grad_(True)
            flow_raw, coords_deform = self.deformation_network(coords_deform, clone=True)
            flow = torch.tanh(flow_raw) * flow_scale

            sampling_grid_flat = sampling_grid.view(-1, 3)
            flow_add = flow + sampling_grid_flat

            deformed_grid = flow_add.view(1, 1, img_size, img_size, 3)
            moved_image_deform = sample_volume_with_grid(moving_image, deformed_grid, mode="bilinear", padding_mode="zeros")

            viz_images["moved_affine"] = moved_image_affine
            viz_images["moved_deform"] = moved_image_deform
            viz_images["flow"] = flow

        return viz_images, sampling_grid_flat, coords_deform


    def compute_losses(self, epoch, viz_images, sampling_grid_flat, coords_deform):
        """
        Compute all registration losses and weighted total.

        Args:
            epoch: The current epoch number
            viz_images: A dictionary with the images used for visualization
            sampling_grid_flat: [H*W, 3] sampling grid (flattened)
            coords_deform: Coords for deformation network, or None in phase 1

        Returns:
            loss_all: Scalar total loss
            tensor_losses: Dict of per-pixel/per-point loss tensors (for visualization)
            scalar_losses: Dict of scalar values for each loss (for tracking/plotting)
            alpha: List of alpha weights for current epoch
        """
        device = self.device
        img_size = self.img_size
        mask = self.mask
        coords = self.coords
        moved_image = viz_images["moved_deform"]
        recon_image = viz_images["reconstructed"]
        support_image = viz_images["support"]
        flow = viz_images["flow"]

        loss_all = torch.tensor(0.0, device=device)
        tensor_losses = {}

        # MSE loss
        loss_mse_tensor = self.losses.mse(self.fixed_image, recon_image)
        loss_mse = loss_mse_tensor.mean()
        tensor_losses["mse"] = loss_mse_tensor

        # Grayscale loss
        loss_grayscale_tensor = self.losses.grayscale(support_image)
        if mask is not None:
            loss_grayscale = loss_grayscale_tensor.sum() / mask.sum().clamp(min=1)
        else:
            loss_grayscale = loss_grayscale_tensor.mean()
        tensor_losses["grayscale"] = loss_grayscale_tensor

        # Background variance loss
        loss_bg_var_tensor = self.losses.bg_var(support_image, mask=mask)
        loss_bg_var = loss_bg_var_tensor.mean()
        tensor_losses["bg_var"] = loss_bg_var_tensor

        # Edge matching loss
        loss_edge_tensor = self.losses.edge(support_image, self.fixed_image, mask=mask)
        loss_edge = loss_edge_tensor.mean()
        tensor_losses["edge"] = loss_edge_tensor

        # NCC losses
        loss_global_ncc = self.losses.ncc(moved_image, self.fixed_image)
        loss_lncc_tensor = self.losses.lncc(moved_image, self.fixed_image, mask=mask)
        loss_lncc = loss_lncc_tensor.mean()
        loss_ncc = (loss_global_ncc + loss_lncc) / 2
        tensor_losses["lncc"] = loss_lncc_tensor

        loss_global_ncc_support = self.losses.ncc(moved_image, support_image)
        loss_lncc_support_tensor = self.losses.lncc(moved_image, support_image, mask=mask)
        loss_lncc_support = loss_lncc_support_tensor.mean()
        loss_ncc_support = (loss_global_ncc_support + loss_lncc_support) / 2
        tensor_losses["lncc_support"] = loss_lncc_support_tensor

        # Deformation losses
        if epoch < self.phase1_epochs:
            # Jacobian loss
            loss_jac_tensor = torch.zeros(coords.shape[0], device=device)
            loss_jac = torch.tensor(0.0, device=device)
            # Flow TV loss
            loss_flow_tv_tensor = torch.zeros(img_size, img_size, device=device)
            loss_flow_tv = torch.tensor(0.0, device=device)
        else:
            # XXX: the coords are 2D so the Jacobian regularization is only applied xy plane
            # TODO: figure out how to apply Jacobian regularization to the z plane as well
            flow_2d = flow[:, :2]
            sampling_grid_2d = sampling_grid_flat[:, :2]
            flow_add_2d = flow_2d + sampling_grid_2d
            loss_jac_tensor = self.losses.reg(coords_deform, flow_add_2d, mask=mask)
            loss_jac = loss_jac_tensor.mean()
            # Flow TV loss
            loss_flow_tv_tensor = self.losses.flow_tv(flow, spatial_size=(img_size, img_size))
            loss_flow_tv = loss_flow_tv_tensor.mean()

        tensor_losses["jacobian"] = loss_jac_tensor
        tensor_losses["flow_tv"] = loss_flow_tv_tensor

        # Alpha weights
        alpha = get_alpha(self.start_alpha, self.end_alpha, epoch, self.total_epochs, self.phase1_epochs)

        if alpha[0] > 0:
            loss_all = loss_all + alpha[0] * loss_mse
        if alpha[1] > 0:
            loss_all = loss_all + alpha[1] * loss_grayscale
        if alpha[2] > 0:
            loss_all = loss_all + alpha[2] * loss_bg_var
        if alpha[3] > 0:
            loss_all = loss_all + alpha[3] * loss_edge
        if alpha[4] > 0:
            loss_all = loss_all + alpha[4] * loss_ncc
        if alpha[5] > 0:
            loss_all = loss_all + alpha[5] * loss_ncc_support
        if alpha[6] > 0:
            loss_all = loss_all + alpha[6] * loss_flow_tv
        if alpha[7] > 0:
            loss_all = loss_all + alpha[7] * loss_jac

        scalar_losses = {
            "lcc": loss_ncc.item(),
            "lcc_support": loss_ncc_support.item(),
            "reconstruction": loss_mse.item(),
            "grayscale": loss_grayscale.item(),
            "edge": loss_edge.item(),
            "bg_var": loss_bg_var.item(),
            "jacobian": loss_jac.item(),
            "flow_tv": loss_flow_tv.item(),
            "lncc": loss_lncc.item(),
            "global_ncc": loss_global_ncc.item(),
            "lncc_support": loss_lncc_support.item(),
            "global_ncc_support": loss_global_ncc_support.item(),
            "total": loss_all.item(),
        }

        return loss_all, tensor_losses, scalar_losses, alpha


def save_affine_params(A, b, output_path, epoch):
    """
    Write affine transformation parameters A and b to a text file.

    Args:
        A: 3x3 affine matrix (torch Parameter or tensor)
        b: 3-element translation vector (torch Parameter or tensor)
        output_path: Base output path (e.g. ending in .png)
        epoch: Epoch number for the filename
    """
    affine_output_path = output_path.replace(".png", f"_{epoch}_affine_params.txt")
    with open(affine_output_path, "w") as f:
        f.write("Affine Transformation Parameters\n")
        f.write(f"Epoch: {epoch}\n")
        f.write("=" * 40 + "\n\n")
        f.write("Matrix A (3x3):\n")
        A_np = A.data.cpu().numpy()
        f.write(f"{A_np[0, 0]:.8f}  {A_np[0, 1]:.8f}  {A_np[0, 2]:.8f}\n")
        f.write(f"{A_np[1, 0]:.8f}  {A_np[1, 1]:.8f}  {A_np[1, 2]:.8f}\n")
        f.write(f"{A_np[2, 0]:.8f}  {A_np[2, 1]:.8f}  {A_np[2, 2]:.8f}\n\n")
        f.write("Translation vector b (3x1):\n")
        b_np = b.data.cpu().numpy()
        f.write(f"{b_np[0]:.8f}\n")
        f.write(f"{b_np[1]:.8f}\n")
        f.write(f"{b_np[2]:.8f}\n\n")
        f.write("Matrix A (Python format):\n")
        f.write(f"[[{A_np[0, 0]:.8f}, {A_np[0, 1]:.8f}, {A_np[0, 2]:.8f}],\n")
        f.write(f" [{A_np[1, 0]:.8f}, {A_np[1, 1]:.8f}, {A_np[1, 2]:.8f}],\n")
        f.write(f" [{A_np[2, 0]:.8f}, {A_np[2, 1]:.8f}, {A_np[2, 2]:.8f}]]\n\n")
        f.write("Translation vector b (Python format):\n")
        f.write(f"[{b_np[0]:.8f}, {b_np[1]:.8f}, {b_np[2]:.8f}]\n")
    return affine_output_path


def build_checkpoint_dict(
    epoch,
    A,
    b,
    support_network,
    residual_network,
    deformation_network,
    network_size,
    n_channels,
    img_size,
    flow_scale,
):
    """
    Build a checkpoint dictionary for torch.save.

    Returns:
        dict suitable for torch.save
    """
    return {
        "epoch": epoch,
        "affine_A": A.data.cpu() if A is not None else None,
        "affine_b": b.data.cpu() if b is not None else None,
        "support_network_state_dict": support_network.state_dict(),
        "residual_network_state_dict": residual_network.state_dict(),
        "deformation_network_state_dict": deformation_network.state_dict(),
        "network_size": network_size,
        "n_channels": n_channels,
        "img_size": img_size,
        "flow_scale": flow_scale,
    }
