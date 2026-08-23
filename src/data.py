import SimpleITK as sitk
import torch
import torch.nn.functional as F

def make_image_3channel(t_img):
    """
    Convert a single channel image to a 3 channel image.
    """
    if t_img.shape[1] == 1:
        t_img = torch.cat([t_img, t_img, t_img], dim=1)

    return t_img


def _compute_target_spatial_shape(spatial_shape, target_size):
    """
    Compute target spatial shape for resize, preserving the minimum (coronal) dimension.
    """
    # Assume the minimum dimension is the coronal dimension
    # We want to keep that intact for 3D MRIs and reshape the image in that plane
    min_val = min(spatial_shape)
    min_dim_idx = spatial_shape.index(min_val)

    # Build the target spatial shape
    new_spatial_shape = []
    for i, dim in enumerate(spatial_shape):
        if i == min_dim_idx:
            new_spatial_shape.append(dim)
        else:
            new_spatial_shape.append(target_size)
    return new_spatial_shape


def _resize_tensor_trilinear(t_img, new_spatial_shape):
    """
    Resize tensor using trilinear interpolation, handling 5D requirement for F.interpolate.
    """
    # Handle 5D requirement for trilinear interpolation
    # F.interpolate needs [Batch, Channel, Depth, Height, Width]
    current_ndims = t_img.dim()

    if current_ndims == 3:      # [x, y, z] -> [1, 1, x, y, z]
        t_img = t_img.unsqueeze(0).unsqueeze(0)
    elif current_ndims == 4:    # [c, x, y, z] -> [1, c, x, y, z]
        t_img = t_img.unsqueeze(0)
    # If it's already 5D, we do nothing.

    # Resize
    t_img = F.interpolate(t_img, size=new_spatial_shape, mode='trilinear', align_corners=False)

    # Bring it back to the original number of dimensions (remove the padding dims)
    while t_img.dim() > current_ndims:
        t_img = t_img.squeeze(0)

    return t_img


def _normalize_tensor_for_inr(t_img, mri=False):
    """
    Normalize tensor to [0,1], optionally set MRI background to white, then scale to [-1, 1].
    """
    # Normalize the image
    t_min, t_max = t_img.min(), t_img.max()
    if t_max > t_min:
        t_img = (t_img - t_min) / (t_max - t_min)

    # If the image is an MRI, make the background white
    if mri:
        t_img[t_img <= 0.1] = 1

    # Convert to range [-1, 1]
    t_img = t_img * 2 - 1

    return t_img


def load_and_preprocess_niftii_image(path, target_size=256, mri=False):
    """
    Load a NIfTI image and preprocess it to a tensor.

    Args:
        path: Path to the NIfTI image
        target_size: Target size of the image
        mri: Whether the image is an MRI

    Returns:
        t_img: Preprocessed tensor image
    """
    # Load the NIfTI image
    sitk_img = sitk.ReadImage(path)
    t_img = convert_sitk_image_to_pytorch_tensor(sitk_img, dtype=torch.float32)

    # Identify the spatial dimensions dynamically
    # We assume the LAST 3 dimensions are x, y, z
    orig_shape = list(t_img.shape)
    spatial_shape = orig_shape[-3:]

    new_spatial_shape = _compute_target_spatial_shape(spatial_shape, target_size)
    t_img = _resize_tensor_trilinear(t_img, new_spatial_shape)
    t_img = _normalize_tensor_for_inr(t_img, mri=mri)

    return t_img


def load_registration_images(moving_path, fixed_path, img_size, device, mri=False):
    """
    Load moving and fixed images for registration, ensure 3-channel, and move to device.

    Args:
        moving_path: Path to the moving image
        fixed_path: Path to the fixed image
        img_size: Target size for preprocessing (passed to load_and_preprocess_niftii_image)
        device: Device to place tensors on
        mri: Whether the moving image is an MRI

    Returns:
        moving_image: Tensor [1, 3, D, H, W] or [1, 3, H, W] on device
        fixed_image: Tensor [1, 3, D, H, W] or [1, 3, H, W] on device
    """
    moving = load_and_preprocess_niftii_image(moving_path, target_size=img_size, mri=mri)
    moving = moving.to(device)
    fixed = load_and_preprocess_niftii_image(fixed_path, target_size=img_size, mri=False)
    fixed = fixed.to(device)

    moving_image = make_image_3channel(moving)
    fixed_image = make_image_3channel(fixed)

    return moving_image, fixed_image


def load_mask_for_registration(mask_path, img_shape, device):
    """
    Load a mask image for registration, convert to tensor, and resize to img_shape.

    Args:
        mask_path: Path to the mask image
        img_shape: Target spatial shape (H, W) for the mask
        device: Device to place tensor on

    Returns:
        mask: Tensor [1, 1, H, W] on device, resized to img_shape
    """
    mask = sitk.ReadImage(mask_path)
    mask = convert_sitk_image_to_pytorch_tensor(mask, device=device)
    mask = F.interpolate(mask, size=img_shape, mode='nearest')
    
    return mask


def convert_sitk_image_to_pytorch_tensor(sitk_img, is_warp=False, **kwargs):
    """
    Convert a SimpleITK image to a PyTorch tensor.

    Args:
        img: SimpleITK image
        is_warp: Whether the image is a warp
        **kwargs: Additional arguments for torch.tensor

    Returns:
        t: PyTorch tensor

    Notes
    -----
    Shape conversions:
    - 2D images: simpleitk [x, y] or [x, y, z] -> [1, 1, x, y] or [1, z, x, y]
    - 2D warp: [x, y, 2] -> [1, x, y, 2]
    - 3D images: simpleitk [z, x, y, c] or [z, x, y] -> [1, c, z, x, y] or [1, 1, z, x, y]
    - 3D warp: [z, x, y, 3] -> [1, z, x, y, 3]
    """
    img_array = sitk.GetArrayFromImage(sitk_img)
    if img_array.shape[0] == 1:
        img_array = img_array.squeeze(0)
    num_components = sitk.Image.GetNumberOfComponentsPerPixel(sitk_img)

    t = torch.tensor(img_array, **kwargs)

    if is_warp:
        t = t.unsqueeze(0) # [1, z, x, y, 3]
        return t

    if num_components > 1:
        # Multi channel image
        dims = list(range(t.ndim))
        new_order = [dims[-1]] + dims[:-1] # move the last dimension to the first position
        t = t.permute(*new_order) # [c, z, x, y] or [c, x, y] handles both 2D and 3D images

    if num_components == 1:
        t = t.unsqueeze(0).unsqueeze(0) # [1, 1, spatial_dims]
    else:
        t = t.unsqueeze(0) # [1, C, spatial_dims]

    return t
