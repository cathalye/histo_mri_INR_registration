import numpy as np
import torch
import SimpleITK as sitk
import torch.nn.functional as F


def get_normalized_coordinates_tensor(img_shape=(256, 256), device='cpu', is_vector=True):
    """
    Create a tensor of normalized coordinates in the range [-1, 1] for the given dimensions.

    Args:
        img_shape: Tuple of integers representing the shape of the image
        is_vector: Whether to return the coordinates as a vector

    Returns:
        coords: Tensor of normalized coordinates in the range [-1, 1]

    Notes:
    -----
    Modification of the method from https://proceedings.mlr.press/v172/wolterink22a.html
    """
    n_dims = len(img_shape)

    coords = [torch.linspace(-1, 1, img_shape[i]) for i in range(n_dims)]
    coords = torch.meshgrid(*coords)
    coords = torch.stack(coords, dim=n_dims)

    if is_vector==True:
        coords = coords.view([np.prod(img_shape), n_dims])

    return coords.to(device=device)


def load_affine_init(transform_path, fixed_img_path, moving_img_path):
    """
    Load affine transform from Greedy format and convert to PyTorch-compatible numpy arrays.

    Args:
        transform_path: Path to the Greedy transform file
        fixed_img_path: Path to the fixed image
        moving_img_path: Path to the moving image

    Returns:
        A_init: 3x3 numpy array (affine matrix)
        b_init: 3-element numpy array (translation vector)
    """
    transform_init = convert_greedy_transform_to_sitk(transform_path)

    fixed_sitk = sitk.ReadImage(fixed_img_path)
    moving_sitk = sitk.ReadImage(moving_img_path)

    A_init, b_init = convert_sitk_affine_transform_to_pytorch(transform_init, fixed_sitk, moving_sitk)

    return A_init, b_init


def convert_greedy_transform_to_sitk(path_transform):
    """
    Load an affine transform from the matrix format used by greedy (and ITK-SNAP) and
    convert it to a SimpleITK AffineTransform object.

    Args:
        path_transform: The path to the transform file
    Returns:
        SimpleITK AffineTransform object

    Notes:
    -----
        - SITK uses LPS coordinate system (left-posterior-superior)
        - c3d, greedy, ITK-SNAP use RAS coordinate system (right-anterior-superior)
        - The signs of x, y coordinate are flipped between LPS and RAS conventions.
    """
    # Load the matrix and convert from RAS to LPS coordinate system
    affine = np.loadtxt(path_transform)
    ras_to_lps = np.diag([-1,-1,1]) # Change of basis
    A = ras_to_lps @ affine[:3,:3] @ ras_to_lps
    b = ras_to_lps @ affine[:3,3]

    # Create an ITK transform
    return sitk.AffineTransform(A.flatten().tolist(), b.tolist())


def get_sitk_voxel_to_physical_transform(image):
    """
    Get the transform from voxel space to physical space for an image.

    Args:
        image: SimpleITK Image
    Returns:
        A, b = get_sitk_voxel_to_physical_transform(image) returns a 3x3 matrix A and a 3x1 vector b such that the physical coordinates
        of a point with voxel coordinates x_voxel can be computed as
        x_phys = A @ x_voxel + b
    """
    A = np.array(image.GetDirection()).reshape(3,3) @ np.diag(image.GetSpacing())
    b = np.array(image.GetOrigin())

    return A, b


def convert_sitk_affine_transform_to_pytorch(tran, img_fix, img_mov):
    """
    Given a fixed and moving image and a SimpleITK affine transform (mov -> fix),
    find the PyTorch affine matrix and translation vector that achieve the same result
    on the normalized [-1, 1] PyTorch coordinates.

    Args:
        tran: A SimpleITK Transform
        img_fix: The fixed image (SimpleITK Image)
        img_mov: The moving image (SimpleITK Image)
    Output:
        A, b = convert_sitk_affine_transform_to_pytorch(tran, img_fix, img_mov)

        A is the 3x3 affine matrix, b is the 3x1 translation vector that achieve
        the same result in PyTorch normalized coordinates space [-1, 1] as
        sitk.Resample(img_mov, img_fix, tran)

    Notes:
    -----
    If using niftii histology images, the 2D image is stored as a 3D image with
    a singleton last dimension. So we can use the convert_sitk_affine_transform_to_pytorch function.

    The logic of the code:
        1. Undo the coordinate normalization norm2vox_fix (fixed)
        2. Convert the voxel coordinates to physical space vox2phys_fix (fixed)
        3. Apply the SITK transform on the physical coordinates - maps physical point
        in fixed image to a physical point in the moving image
        4. Convert the physical coordinates to voxel coordinates phys2vox_mov (moving)
        5. Convert the voxel coordinates to normalized coordinates vox2norm_mov (moving)
    """

    A_vox2phys_fix, b_vox2phys_fix = get_sitk_voxel_to_physical_transform(img_fix)
    A_vox2phys_mov, b_vox2phys_mov = get_sitk_voxel_to_physical_transform(img_mov)

    # Pad the 2D Histology matrix to 3x3 if it's 2x2
    if A_vox2phys_fix.shape == (2, 2):
        A_vox2phys_fix_2d = A_vox2phys_fix
        A_vox2phys_fix = np.eye(3)
        A_vox2phys_fix[:2, :2] = A_vox2phys_fix_2d
        # We need a dummy size for the 3rd dimension
        size_fix = np.array([*img_fix.GetSize(), 1])
    else:
        size_fix = np.array(img_fix.GetSize())

    A_sitk = np.array(tran.GetMatrix()).reshape(3,3)
    b_sitk = np.array(tran.GetTranslation())

    # Get the mapping from voxel index to PyTorch normalized [-1, 1] coordinates
    size_mov = np.array(img_mov.GetSize())

    A_vox2norm_fix = np.diag(2.0 / size_fix)
    A_vox2norm_mov = np.diag(2.0 / size_mov)

    b_vox2norm_fix = 1.0 / size_fix - 1.0
    b_vox2norm_mov = 1.0 / size_mov - 1.0

    # Compute inverses needed for the composed transformtggf
    A_norm2vox_fix = np.linalg.inv(A_vox2norm_fix)
    A_phys2vox_mov = np.linalg.inv(A_vox2phys_mov)

    # Compose the full transform: PyTorch norm fixed coords -> PyTorch norm moving coords
    # norm_fix -> vox_fix -> phys_fix -> (affine transform) -> phys_mov -> vox_mov -> norm_mov
    A = A_vox2norm_mov @ A_phys2vox_mov @ A_sitk @ A_vox2phys_fix @ A_norm2vox_fix # Chain of matrices in reverse order
    # XXX: do this calculation by hand to understand the conversion
    b = A_vox2norm_mov @ A_phys2vox_mov @ (A_sitk @ b_vox2phys_fix + b_sitk - b_vox2phys_mov) - A @ b_vox2norm_fix + b_vox2norm_mov

    # Return the final transform
    return A, b


def sample_volume_with_grid(T_mov, sampling_grid, mode='bilinear', padding_mode='zeros'):
    """
    Sample a 3D volume using a custom sampling grid (e.g., deformed coordinates).

    Args:
        T_mov: 3D moving volume tensor [C, D, H, W] or [N, C, D, H, W]
        sampling_grid: Sampling grid tensor [1, 1, H, W, 3] with coordinates in [-1, 1]
        mode: Interpolation mode ('bilinear' or 'nearest')
        padding_mode: Padding mode ('zeros', 'border', or 'reflection')

    Returns:
        T_resampled: Sampled 2D image [C, H, W]

    Notes:
    -----
    F.grid_sample - computes output values for an input tensor at grid points (output from affine_grid)
                by interpolation

    See reference https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
    """
    # Ensure T_mov is 5D: (N, C, D, H, W)
    if T_mov.dim() == 4:
        T_mov = T_mov.unsqueeze(0)

    # Sample the volume
    T_resampled = F.grid_sample(T_mov, sampling_grid, mode=mode, padding_mode=padding_mode, align_corners=False)

    # Remove the Batch and singleton Depth dimensions
    # Resulting shape: (C, H, W)
    return T_resampled.squeeze(0).squeeze(1)


def apply_affine_transform_pytorch(T_ref, T_mov, A, b, mode='bilinear', padding_mode='zeros'):
    """
    Apply an affine transform to images of same dimensions. All inputs are pytorch tensors.

    Args:
        T_ref: The reference image (tensor)
        T_mov: The moving image (tensor)
        A: The affine transform (tensor)
        b: The translation vector (tensor)
        mode: The interpolation mode
        padding_mode: The padding mode
    Returns:
        The transformed image (tensor)

    Notes:
    -----
    This function operates on images of the same dimensions and is a simplified
    version of the 2d3d function below.

    F.affine_grid - generates a 2D or 3D flow field (sampling grid) given an affine
                    transformation matrix and the size of the output grid
    F.grid_sample - computes output values for an input tensor at grid points (output from affine_grid)
                    by interpolation

    See references:
    https://pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html
    https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
    """

    if T_ref.dim() != T_mov.dim():
        raise ValueError(f"T_ref and T_mov must have the same number of dimensions, but got {T_ref.dim()} and {T_mov.dim()} \n \
            Use apply_affine_transform_pytorch_2d3d for 2D to 3D transformation")

    T_ref = T_ref.unsqueeze(0)

    affine_matrix = torch.cat([A, b.unsqueeze(1)], dim=1)

    grid = F.affine_grid(affine_matrix.unsqueeze(0), size=T_ref.shape, align_corners=False)

    T_resampled = F.grid_sample(T_mov, grid, mode=mode, padding_mode=padding_mode, align_corners=False)

    return grid, T_resampled


def apply_affine_transform_pytorch_2d3d(T_ref, T_mov, A, b, mode='bilinear', padding_mode='zeros'):
    """
    Apply an affine transform to a 3D volume that extracts a 2D slice from the 3D volume.
    All inputs are pytorch tensors.

    Args:
        T_ref: 2D reference tensor [C, H, W] (used for output shape)
        T_mov: 3D moving volume tensor [C, D, H, W]
        A: Normalized 3x3 affine matrix
        b: Normalized 3x1 translation vector

    Notes:
    -----
    When mode='bilinear' and the input is 5-D, the interpolation mode used
    internally will actually be trilinear.

    See the apply_affine_transform_pytorch function above for the details.
    """

    if T_ref.dim() == T_mov.dim():
        raise ValueError(f"T_ref and T_mov must have different number of dimensions, but got {T_ref.dim()} and {T_mov.dim()} \n \
            Use apply_affine_transform_pytorch for same dimension transformation")

    device = T_mov.device
    dtype = T_mov.dtype

    # Ensure T_mov is 5D
    if T_mov.dim() == 4:
        T_mov = T_mov.unsqueeze(0)

    # Get dimensions of the 2D reference image
    # Note: T_ref shape is usually (C, H, W) or (H, W)
    H, W = T_ref.shape[-2:]

    # Build the affine matrix [A | b]
    A = A.to(device=device, dtype=dtype)
    b = b.to(device=device, dtype=dtype)
    affine_matrix = torch.cat([A, b.unsqueeze(1)], dim=1).unsqueeze(0) # Shape (1, 3, 4)

    # Output size (N, C, D_out=1, H_out, W_out)
    # Note: D_out=1 because we are extracting exactly one slice
    output_size = (1, T_mov.shape[1], 1, H, W)

    sampling_grid = F.affine_grid(affine_matrix, output_size, align_corners=False)

    # Sample volume with the grid
    T_resampled = F.grid_sample(T_mov, sampling_grid, mode=mode, padding_mode=padding_mode, align_corners=False)

    # Remove the Batch and singleton Depth dimensions
    # Resulting shape: (1, 1, H, W)
    return sampling_grid, T_resampled.squeeze(0).squeeze(1)
