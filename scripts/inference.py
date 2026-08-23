import torch
import numpy as np
import argparse
import os
import sys
import SimpleITK as sitk

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from networks import ImpNet
from data import make_coords_tensor, sitk_to_pytorch
from transforms import transform_image_pytorch_2d3d, sample_volume_with_grid


def load_checkpoint(checkpoint_path, device):
    """Load a saved checkpoint and reconstruct the networks."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    network_size = checkpoint['network_size']
    n_channels = checkpoint['n_channels']
    flow_scale = checkpoint['flow_scale']
    img_size = checkpoint['img_size']

    support_network = ImpNet(
        input_dim=2, output_dim=n_channels, hidden_dim=256, hidden_n_layers=network_size[1],
        last_layer_weights_small=False, is_last_linear=True, input_omega_0=30.0,
        hidden_omega_0=30., input_n_encoding_functions=6
    ).to(device)

    residual_network = ImpNet(
        input_dim=2, output_dim=n_channels, hidden_dim=256, hidden_n_layers=network_size[1],
        last_layer_weights_small=False, is_last_linear=True, input_omega_0=30.0,
        hidden_omega_0=30., input_n_encoding_functions=6
    ).to(device)

    deformation_network = ImpNet(
        input_dim=2, output_dim=3, hidden_dim=256, hidden_n_layers=network_size[0],
        last_layer_weights_small=True, is_last_linear=True, input_omega_0=5.0,
        hidden_omega_0=5.0, input_n_encoding_functions=0
    ).to(device)

    support_network.load_state_dict(checkpoint['support_network_state_dict'])
    residual_network.load_state_dict(checkpoint['residual_network_state_dict'])
    deformation_network.load_state_dict(checkpoint['deformation_network_state_dict'])

    support_network.eval()
    residual_network.eval()
    deformation_network.eval()

    affine_A = checkpoint['affine_A'].to(device) if checkpoint['affine_A'] is not None else None
    affine_b = checkpoint['affine_b'].to(device) if checkpoint['affine_b'] is not None else None

    return {
        'support_network': support_network,
        'residual_network': residual_network,
        'deformation_network': deformation_network,
        'affine_A': affine_A,
        'affine_b': affine_b,
        'flow_scale': flow_scale,
        'n_channels': n_channels,
        'img_size': img_size,
        'epoch': checkpoint['epoch'],
    }


def compute_outputs_at_resolution(models, output_size, device):
    """
    Compute support, residual, and deformation outputs at the specified resolution.

    Args:
        models: Dictionary containing loaded networks and parameters
        output_size: Tuple (H, W) for output resolution
        device: torch device

    Returns:
        Dictionary with support_image, residual_image, reconstructed_image, flow
    """
    h, w = output_size
    n_channels = models['n_channels']
    flow_scale = models['flow_scale']

    coords = make_coords_tensor(img_shape=(h, w))
    coords = coords.to(device)

    with torch.no_grad():
        support_pixels = models['support_network'](coords, clone=False)
        support_image = support_pixels.view(h, w, n_channels).permute(2, 0, 1).unsqueeze(0)

        residual_pixels = models['residual_network'](coords, clone=False)
        residual_image = residual_pixels.view(h, w, n_channels).permute(2, 0, 1).unsqueeze(0)

        reconstructed_image = support_image + residual_image

        flow_raw = models['deformation_network'](coords, clone=False)
        flow = torch.tanh(flow_raw) * flow_scale
        flow_image = flow.view(h, w, 3)

    return {
        'support': support_image,
        'residual': residual_image,
        'reconstructed': reconstructed_image,
        'flow': flow,
        'flow_image': flow_image,
    }


def apply_deformation_to_mri(mri_volume, models, fixed_image_for_grid, output_size, device):
    """
    Apply the learned affine + deformation to sample the MRI volume at the specified resolution.

    Args:
        mri_volume: 3D MRI volume tensor [C, D, H, W] or [N, C, D, H, W]
        models: Dictionary containing loaded networks and parameters
        fixed_image_for_grid: Fixed image tensor used to define the 2D output space
        output_size: Tuple (H, W) for output resolution
        device: torch device

    Returns:
        moved_affine: MRI slice after affine transform only
        moved_deform: MRI slice after affine + deformation
        sampling_grid: The affine sampling grid
        flow_image: Deformation flow field [H, W, 3]
    """
    h, w = output_size
    flow_scale = models['flow_scale']
    A = models['affine_A']
    b = models['affine_b']

    coords_init = make_coords_tensor(img_shape=(h, w))
    coords_init = coords_init.to(device)

    with torch.no_grad():
        sampling_grid, moved_affine = transform_image_pytorch_2d3d(
            fixed_image_for_grid, mri_volume, A, b,
            padding_mode='zeros'
        )

        flow_raw = models['deformation_network'](coords_init, clone=False)
        flow = torch.tanh(flow_raw) * flow_scale

        sampling_grid_flat = sampling_grid.view(-1, 3)
        flow_add = flow + sampling_grid_flat

        deformed_grid = flow_add.view(1, 1, h, w, 3)
        moved_deform = sample_volume_with_grid(mri_volume, deformed_grid, mode='bilinear', padding_mode='zeros')
        flow_image = flow.view(h, w, 3)

    return moved_affine, moved_deform, sampling_grid, flow_image


def save_image_as_nifti(image_tensor, output_path, reference_image=None):
    """Save a 2D image tensor as a NIfTI file."""
    image_np = image_tensor.squeeze().cpu().numpy()

    if image_np.ndim == 3:
        image_np = np.transpose(image_np, (1, 2, 0))

    image_sitk = sitk.GetImageFromArray(image_np, isVector=(image_np.ndim == 3))

    if reference_image is not None:
        spacing = list(reference_image.GetSpacing())
        if len(spacing) == 3:
            spacing = spacing[:2]
        image_sitk.SetSpacing(spacing)

    sitk.WriteImage(image_sitk, output_path)
    print(f"Saved: {output_path}")


def save_flow_as_nifti(flow_tensor, output_path):
    """Save the flow field as a NIfTI file."""
    flow_np = flow_tensor.cpu().numpy()

    flow_sitk = sitk.GetImageFromArray(flow_np, isVector=True)
    sitk.WriteImage(flow_sitk, output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply trained networks to full-resolution MRI.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the saved checkpoint (.pt file)')
    parser.add_argument('--mri', type=str, required=True, help='Path to the full-resolution MRI volume')
    parser.add_argument('--fixed', type=str, required=True, help='Path to the fixed (histology) image for grid definition')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output images')
    parser.add_argument('--output_size', type=int, nargs=2, default=None, help='Output resolution (H W). If not specified, uses fixed image size.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--save_nifti', action='store_true', help='Save outputs as NIfTI files')
    parser.add_argument('--save_numpy', action='store_true', help='Save outputs as NumPy files')
    parser.add_argument('--prefix', type=str, default=None, help='Custom prefix for output files (default: checkpoint filename)')
    parser.add_argument('--contour', action='store_true', help='Only compute affine and deformation transformed images (no support, residual, reconstructed)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading checkpoint from: {args.checkpoint}")
    models = load_checkpoint(args.checkpoint, device)
    print(f"Loaded checkpoint from epoch {models['epoch']}")
    print(f"  Network trained at resolution: {models['img_size']}x{models['img_size']}")
    print(f"  Flow scale: {models['flow_scale']}")
    print(f"  Channels: {models['n_channels']}")

    print(f"Loading MRI volume from: {args.mri}")
    mri_sitk = sitk.ReadImage(args.mri)
    mri_tensor = sitk_to_pytorch(mri_sitk, dtype=torch.float32).to(device)

    mri_min, mri_max = mri_tensor.min(), mri_tensor.max()
    if mri_max > mri_min:
        mri_tensor = (mri_tensor - mri_min) / (mri_max - mri_min)
    mri_tensor = mri_tensor * 2 - 1

    print(f"  MRI shape: {mri_tensor.shape}")

    print(f"Loading fixed image from: {args.fixed}")
    fixed_sitk = sitk.ReadImage(args.fixed)
    fixed_tensor = sitk_to_pytorch(fixed_sitk, dtype=torch.float32).to(device)

    fixed_min, fixed_max = fixed_tensor.min(), fixed_tensor.max()
    if fixed_max > fixed_min:
        fixed_tensor = (fixed_tensor - fixed_min) / (fixed_max - fixed_min)
    fixed_tensor = fixed_tensor * 2 - 1

    print(f"  Fixed image shape: {fixed_tensor.shape}")

    if args.output_size is not None:
        output_size = tuple(args.output_size)
    else:
        output_size = (fixed_tensor.shape[-2], fixed_tensor.shape[-1])
    print(f"Output resolution: {output_size}")

    if not args.contour:
        print("\nComputing support, residual, and deformation at output resolution...")
        outputs = compute_outputs_at_resolution(models, output_size, device)

    print("\nApplying affine + deformation to MRI volume...")
    moved_affine, moved_deform, _, flow_image = apply_deformation_to_mri(
        mri_tensor, models, fixed_tensor, output_size, device
    )

    if args.prefix is not None:
        output_prefix = args.prefix + "_"
    else:
        output_prefix = ""

    if args.save_nifti:
        moved_affine_01 = (moved_affine + 1) / 2
        moved_deform_01 = (moved_deform + 1) / 2

        save_image_as_nifti(moved_affine_01, os.path.join(args.output_dir, f'{output_prefix}affine_result.nii.gz'), fixed_sitk)
        save_image_as_nifti(moved_deform_01, os.path.join(args.output_dir, f'{output_prefix}deformable_result.nii.gz'), fixed_sitk)


        if not args.contour:
            support_01 = (outputs['support'] + 1) / 2
            residual_01 = (outputs['residual'] + 1) / 2
            reconstructed_01 = (outputs['reconstructed'] + 1) / 2
            save_image_as_nifti(support_01, os.path.join(args.output_dir, f'{output_prefix}support.nii.gz'), fixed_sitk)
            save_image_as_nifti(residual_01, os.path.join(args.output_dir, f'{output_prefix}residual.nii.gz'), fixed_sitk)
            save_image_as_nifti(reconstructed_01, os.path.join(args.output_dir, f'{output_prefix}reconstructed.nii.gz'), fixed_sitk)

            save_flow_as_nifti(flow_image, os.path.join(args.output_dir, f'{output_prefix}flow.nii.gz'))

    if args.save_numpy or (not args.save_nifti):
        np.save(os.path.join(args.output_dir, f'{output_prefix}flow.npy'),
                flow_image.cpu().numpy())
        np.save(os.path.join(args.output_dir, f'{output_prefix}moved_affine.npy'),
                moved_affine.cpu().numpy())
        np.save(os.path.join(args.output_dir, f'{output_prefix}moved_deform.npy'),
                moved_deform.cpu().numpy())
        np.savez(os.path.join(args.output_dir, f'{output_prefix}affine_params.npz'),
                 A=models['affine_A'].cpu().numpy(),
                 b=models['affine_b'].cpu().numpy())

        if not args.contour:
            np.save(os.path.join(args.output_dir, f'{output_prefix}support.npy'),
                    outputs['support'].cpu().numpy())
            np.save(os.path.join(args.output_dir, f'{output_prefix}residual.npy'),
                    outputs['residual'].cpu().numpy())
            np.save(os.path.join(args.output_dir, f'{output_prefix}reconstructed.npy'),
                    outputs['reconstructed'].cpu().numpy())

        print(f"\nNumPy files saved to: {args.output_dir}")

    print("\nInference complete!")
    flow_flat = flow_image.view(-1, 3)
    print(f"  Flow magnitude range: [{flow_flat.abs().min():.6f}, {flow_flat.abs().max():.6f}]")
    print(f"  Moved (affine) range: [{moved_affine.min():.4f}, {moved_affine.max():.4f}]")
    print(f"  Moved (deform) range: [{moved_deform.min():.4f}, {moved_deform.max():.4f}]")
    if not args.contour:
        print(f"  Support image range: [{outputs['support'].min():.4f}, {outputs['support'].max():.4f}]")
        print(f"  Residual image range: [{outputs['residual'].min():.4f}, {outputs['residual'].max():.4f}]")
