import torch
from tqdm import tqdm

import argparse
import gc
import os
import sys
import SimpleITK as sitk

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from networks import prepare_networks, set_seeds, create_phase2_optimizer, freeze_affine_parameters
from losses import RegistrationLosses
from data import load_registration_images, load_mask_for_registration
from training_logs import MetricMonitor, Visualization
from transforms import get_normalized_coordinates_tensor, apply_affine_transform_pytorch_2d3d, load_affine_init
from registration import RegistrationConfig, RegistrationTrainer, save_affine_params, build_checkpoint_dict
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register two images and save the result visualization.")
    parser.add_argument('--moving', type=str, required=True, help='Path to the moving image')
    parser.add_argument('--fixed', type=str, required=True, help='Path to the fixed image')
    parser.add_argument('--mask', type=str, required=True, help='Path to the mask image')
    parser.add_argument('--mri', type=bool, default=False, help='Whether the fixed image is an MRI')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output visualization')
    parser.add_argument('--img_size', type=int, default=256, help='Size of the images')
    parser.add_argument('--start_alpha', type=float, nargs=8, default=[100, 1, 10, 1, 1, 1, 0, 1], help='Starting weights: mse, grayscale, bg_var, edge, lncc, lncc_support, flow_tv, jacobian')
    parser.add_argument('--end_alpha', type=float, nargs=8, default=[100, 40, 20, 20, 1, 2, 50, 1], help='Ending weights (linearly interpolates from start to end)')
    parser.add_argument('--network_size', type=int, nargs=2, default=[5, 5], help='Number of hidden layers in deformation and support-residual networks')
    parser.add_argument('--epochs', type=int, default=400, help='Total number of training epochs (phase 1 + phase 2)')
    parser.add_argument('--phase1_epochs', type=int, default=200, help='Number of epochs for phase 1 (decomposition + affine)')
    parser.add_argument('--save_at_epochs', type=int, nargs='+', default=[50], help='Save the model at these specific epochs')
    parser.add_argument('--deformation_lr_multiplier', type=float, default=1.0, help='Learning rate multiplier for deformation network in phase 2')
    parser.add_argument('--flow_scale', type=float, default=0.1, help='Maximum flow magnitude (applies tanh * flow_scale to network output)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--affine_lr_multiplier', type=float, default=10.0, help='Learning rate multiplier for affine parameters A and b (default: 10.0)')
    parser.add_argument('--transform_init', type=str, required=True, help='Path to the initial affine transformation parameters')
    args = parser.parse_args()

    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    moving_path = args.moving
    fixed_path = args.fixed
    mask_path = args.mask
    mri = args.mri
    output_path = args.output
    img_size = args.img_size
    network_size = args.network_size
    start_alpha = args.start_alpha
    end_alpha = args.end_alpha
    epochs = args.epochs
    phase1_epochs = args.phase1_epochs
    save_at_epochs = args.save_at_epochs
    device = args.device
    deformation_lr_multiplier = args.deformation_lr_multiplier
    flow_scale = args.flow_scale
    affine_lr_multiplier = args.affine_lr_multiplier
    transform_init = args.transform_init

    A_init, b_init = load_affine_init(transform_init, fixed_path, moving_path)
    # TODO: rethink how the mri flag is used (currently used for making background white if MRI)
    # The utility is only if we are going to use this method for IHC to histo registration as well
    # If we end up using the puple mask to mask the MRI, we won't need the MRI flag
    moving_image, fixed_image = load_registration_images(moving_path, fixed_path, img_size, device, mri=mri)
    mask = load_mask_for_registration(mask_path, (img_size, img_size), device)

    # XXX: hardcoded number of channels for the input images
    # TODO: Have support network output a single channel image
    n_channels = 3

    coords_init = get_normalized_coordinates_tensor(img_shape=(img_size, img_size), device=device)
    coords = coords_init.clone().detach().requires_grad_(True)

    # prepare networks and losses
    set_seeds()
    deformation_network, support_network, residual_network, optimizer, A, b = prepare_networks(network_size, n_channels, device,
                                                                                               affine_transform=True, affine_lr_multiplier=affine_lr_multiplier)
    registration_losses = RegistrationLosses.prepare(device, n_channels)

    config = RegistrationConfig(
        img_size=img_size,
        n_channels=n_channels,
        phase1_epochs=phase1_epochs,
        total_epochs=epochs,
        flow_scale=flow_scale,
        device=device,
        start_alpha=start_alpha,
        end_alpha=end_alpha,
    )

    trainer = RegistrationTrainer(
        config=config,
        losses=registration_losses,
        fixed_image=fixed_image,
        moving_image=moving_image,
        mask=mask,
        coords=coords,
        coords_init=coords_init,
        support_network=support_network,
        residual_network=residual_network,
        deformation_network=deformation_network,
        A=A,
        b=b,
    )

    # Create dictionary to store all images for visualization
    viz_images = {'moving': moving_image, 'fixed': fixed_image}

    stream = tqdm(range(epochs))
    loop_monitor = MetricMonitor()

    losses = {'lcc': [], 'lcc_support': [], 'reconstruction': [], 'grayscale': [], 'edge': [], 'bg_var': [],
              'jacobian': [], 'flow_tv': [], 'global_ncc': [], 'global_ncc_support': [], 'lncc': [], 'lncc_support': [], 'total': []}

    # Initialize A and b with the loaded transform values while keeping them as Parameters
    # This ensures they remain in the computation graph and optimizer
    with torch.no_grad():
        A.data = torch.from_numpy(A_init).to(device, dtype=A.dtype)
        b.data = torch.from_numpy(b_init).to(device, dtype=b.dtype)

    for epoch in stream:
        # Phase transition: at phase1_epochs, freeze affine and create new optimizer
        # TODO: ensure the trainer A, b parameters are getting frozen correctly?
        if epoch == phase1_epochs:
            print(f"\n=== Phase 2 starting at epoch {epoch} ===")
            print("Freezing affine parameters and enabling deformation network...")
            freeze_affine_parameters(A, b)
            optimizer = create_phase2_optimizer(deformation_network, support_network, residual_network,
                                                base_lr=0.0001, deformation_lr_multiplier=deformation_lr_multiplier)

        viz_images, sampling_grid_flat, coords_deform = trainer.step(epoch, viz_images)

        loss_all, tensor_losses, scalar_losses, alpha = trainer.compute_losses(epoch, viz_images, sampling_grid_flat, coords_deform)

        # Track losses for plotting
        for k, v in scalar_losses.items():
            losses[k].append(v)

        alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, alpha8 = alpha
        loop_monitor.update('reconstruction', alpha1 * scalar_losses['reconstruction'])
        loop_monitor.update('grayscale', alpha2 * scalar_losses['grayscale'])
        loop_monitor.update('bg_var', alpha3 * scalar_losses['bg_var'])
        loop_monitor.update('edge', alpha4 * scalar_losses['edge'])
        loop_monitor.update('Lcc', alpha5 * scalar_losses['lcc'])
        loop_monitor.update('Lcc_support', alpha6 * scalar_losses['lcc_support'])
        loop_monitor.update('flow_tv', alpha7 * scalar_losses['flow_tv'])
        loop_monitor.update('jacobian', alpha8 * scalar_losses['jacobian'])
        loop_monitor.update('loss', loss_all.item())
        stream.set_description(f'Epoch: {epoch}. Train. {loop_monitor}')

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        if epoch == 0:
            viz_images['moving'] = viz_images['moved_deform']

        if (epoch+1) in save_at_epochs:
            viz = Visualization(losses, tensor_losses, alpha, epoch+1, viz_images, (img_size, img_size), output_path)
            viz.reconstruct_pixelwise_losses()
            viz.reconstruct_images()
            viz.plot_losses_over_epochs()
            viz.visualize_flow()  # Visualize deformation flow field

            if A is not None and b is not None:
                affine_output_path = save_affine_params(A, b, output_path, epoch + 1)
                print(f"Affine parameters saved to: {affine_output_path}")

            checkpoint_path = output_path.replace('.png', f'_{epoch+1}_checkpoint.pt')
            checkpoint = build_checkpoint_dict(
                epoch=epoch + 1,
                A=A,
                b=b,
                support_network=support_network,
                residual_network=residual_network,
                deformation_network=deformation_network,
                network_size=network_size,
                n_channels=n_channels,
                img_size=img_size,
                flow_scale=flow_scale,
            )
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")

    del deformation_network, support_network, residual_network
    del optimizer, registration_losses
    del viz_images, tensor_losses

    torch.cuda.empty_cache()
    gc.collect()
