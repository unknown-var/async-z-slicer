import trimesh
import numpy as np
from skimage.measure import marching_cubes
from trimesh.voxel import VoxelGrid
from scipy.ndimage import distance_transform_edt

def stl_to_sdf(mesh: trimesh.Trimesh, pitch: float, pad_voxels: int = 10) -> tuple[np.ndarray, VoxelGrid, np.ndarray]:
    """
    Build a signed distance field using inside/outside distance transforms with padding.
    """
    # voxel grid (occupied surface + interior)
    vg = mesh.voxelized(pitch).fill()
    solid = vg.matrix.astype(bool)

    # Pad the solid array with False (empty space) on all axes
    solid_padded = np.pad(solid, pad_width=pad_voxels, mode='constant', constant_values=False)

    # outside = inverse
    outside = ~solid_padded

    # distance transforms
    dist_inside = distance_transform_edt(solid_padded)
    dist_outside = distance_transform_edt(outside)

    sdf = dist_outside - dist_inside
    
    # Calculate the physical translation offset caused by adding padding
    pad_offset = np.array([-pad_voxels * pitch] * 3)
    
    return sdf, vg, pad_offset

def inset_sdf(sdf: np.ndarray, pitch: float, inset_mm: float) -> np.ndarray:
    """
    Apply a uniform inset to the entire geometry.
    """
    inset_vox = inset_mm / pitch
    
    # Adding to the entire SDF shifts the zero-boundary inward (shrinks object)
    return sdf + inset_vox

def sdf_to_mesh(sdf: np.ndarray, pitch: float, vg: VoxelGrid, pad_offset: np.ndarray) -> trimesh.Trimesh:
    """
    Convert SDF back to mesh using marching cubes, adjusting for padding.
    """
    verts, faces, normals, values = marching_cubes(
        sdf,
        level=0,
        spacing=(pitch, pitch, pitch)
    )

    # Apply both the original voxel grid translation and the negative padding offset
    verts += vg.translation + pad_offset

    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def generate_inset_mesh(
    mesh: trimesh.Trimesh, 
    horizontal_offset: float, 
    vertical_offset: float, 
    pitch: float = 0.2
) -> trimesh.Trimesh | None:
    """
    Generates an inner wall mesh using SDF.
    Allows independent horizontal and vertical offsets.
    """
    if horizontal_offset <= 0 and vertical_offset <= 0:
        return mesh.copy()

    # Determine anisotropic scale factor
    z_scale = 1.0
    if vertical_offset > 0 and horizontal_offset > 0:
        z_scale = horizontal_offset / vertical_offset

    # Scale mesh for anisotropic SDF computation
    scaled_mesh = mesh.copy()
    if z_scale != 1.0:
        scaled_mesh.apply_scale([1.0, 1.0, z_scale])

    pad = 10

    sdf, vg, pad_offset = stl_to_sdf(scaled_mesh, pitch, pad_voxels=pad)

    sdf2 = inset_sdf(sdf, pitch, inset_mm=horizontal_offset)


    # Reconstruct the offset mesh
    inset_mesh = sdf_to_mesh(sdf2, pitch, vg, pad_offset)

    # Invert the initial anisotropic Z-scale
    if z_scale != 1.0:
        inset_mesh.apply_scale([1.0, 1.0, 1.0 / z_scale])

    # Laplacian smoothing to reduce surface noise from voxelization
    trimesh.smoothing.filter_laplacian(inset_mesh, iterations=3)

    return inset_mesh
