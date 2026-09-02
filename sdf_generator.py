import trimesh
import numpy as np
import os
from skimage.measure import marching_cubes
from trimesh.voxel import VoxelGrid

# Use multi-threaded edt package if available, fall back to scipy
try:
    import edt as edt_lib
    _num_threads = max(1, os.cpu_count() or 1)
    def _edt(data: np.ndarray) -> np.ndarray:
        """Multi-threaded Euclidean distance transform using the edt package."""
        return edt_lib.edt(data, parallel=_num_threads)
    _edt_backend = f"edt (multi-threaded, {_num_threads} threads)"
except ImportError:
    from scipy.ndimage import distance_transform_edt
    def _edt(data: np.ndarray) -> np.ndarray:
        """Fallback single-threaded EDT using scipy."""
        return distance_transform_edt(data)
    _edt_backend = "scipy (single-threaded)"


def stl_to_sdf(mesh: trimesh.Trimesh, pitch: float, pad_voxels: int = 10) -> tuple[np.ndarray, VoxelGrid, np.ndarray]:
    """
    Build a signed distance field using inside/outside distance transforms with padding.
    """
    # Calculate a safe max_iter for mesh subdivision so fine pitch or large meshes don't hit trimesh's default cap of 10
    try:
        edges = mesh.vertices[mesh.edges[:, 0]] - mesh.vertices[mesh.edges[:, 1]]
        longest_edge = float(np.linalg.norm(edges, axis=1).max())
        max_edge = pitch / 2.0
        # Add safety headroom to prevent premature ValueError: max_iter exceeded!
        max_iter = max(int(np.ceil(np.log2(longest_edge / max_edge))) + 5, 20)
    except Exception:
        max_iter = 30

    # voxel grid (occupied surface + interior)
    try:
        vg = mesh.voxelized(pitch, max_iter=max_iter).fill()
    except ValueError:
        try:
            vg = mesh.voxelized(pitch, max_iter=max_iter + 10).fill()
        except Exception:
            vg = mesh.voxelized(pitch, method='ray').fill()

    solid = vg.matrix.astype(bool)

    # Pad the solid array with False (empty space) on all axes
    solid_padded = np.pad(solid, pad_width=pad_voxels, mode='constant', constant_values=False)

    # outside = inverse
    outside = ~solid_padded

    # distance transforms (uses multi-threaded edt if available)
    dist_inside = _edt(solid_padded)
    dist_outside = _edt(outside)

    sdf = dist_outside - dist_inside
    
    # Calculate the physical translation offset caused by adding padding
    pad_offset = np.array([-pad_voxels * pitch] * 3)
    
    return sdf, vg, pad_offset


class SDFCache:
    """
    Precomputes and caches a Signed Distance Field for a mesh at a given pitch and z_scale.
    
    The z_scale ratio (horizontal_offset / vertical_offset) is constant across all wall
    and infill passes, so the expensive voxelization + EDT computation can be done once.
    Each inset mesh is then generated cheaply by shifting the cached SDF iso-level
    and running marching cubes.
    """
    
    def __init__(self, mesh: trimesh.Trimesh, pitch: float, z_scale: float = 1.0, pad_voxels: int = 10):
        self.pitch = pitch
        self.z_scale = z_scale
        self.pad_voxels = pad_voxels
        
        # Apply anisotropic Z-scale to the mesh
        scaled_mesh = mesh.copy()
        if z_scale != 1.0:
            scaled_mesh.apply_scale([1.0, 1.0, z_scale])
        
        # Compute the full SDF (this is the expensive part: voxelization + 2× EDT)
        self.sdf, self.vg, self.pad_offset = stl_to_sdf(scaled_mesh, pitch, pad_voxels=pad_voxels)
        print(f"  SDF cache built: grid shape={self.sdf.shape}, EDT backend={_edt_backend}")
    
    def generate_inset(self, horizontal_offset: float) -> trimesh.Trimesh | None:
        """
        Generate an inset mesh from the cached SDF.
        Only runs the cheap operations: SDF shift → marching cubes → Z-unscale → Laplacian smooth.
        """
        sdf_shifted = inset_sdf(self.sdf, self.pitch, inset_mm=horizontal_offset)
        
        # Reconstruct mesh from shifted SDF
        inset_mesh = sdf_to_mesh(sdf_shifted, self.pitch, self.vg, self.pad_offset)
        
        # Undo the anisotropic Z-scale
        if self.z_scale != 1.0:
            inset_mesh.apply_scale([1.0, 1.0, 1.0 / self.z_scale])
        
        # Laplacian smoothing to reduce surface noise from voxelization
        trimesh.smoothing.filter_laplacian(inset_mesh, iterations=3)
        
        return inset_mesh


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
    pitch: float = 0.2,
    sdf_cache: SDFCache | None = None
) -> trimesh.Trimesh | None:
    """
    Generates an inner wall mesh using SDF.
    Allows independent horizontal and vertical offsets.

    :param mesh: Input triangle mesh (coordinates in mm).
    :param horizontal_offset: Horizontal inset distance in mm.
    :param vertical_offset: Vertical offset distance in mm.
    :param pitch: Voxel grid resolution / cell size in millimeters (mm).
                 Controls the discretization fineness of the Signed Distance Field.
                 Smaller values yield higher detail at the cost of compute time and RAM.
    :param sdf_cache: Optional precomputed SDF cache. When provided, skips the
                     expensive voxelization and EDT computation and only runs
                     marching cubes + smoothing.
    """
    if horizontal_offset <= 0 and vertical_offset <= 0:
        return mesh.copy()

    # Use cached SDF if available (fast path: only marching cubes + smooth)
    if sdf_cache is not None:
        return sdf_cache.generate_inset(horizontal_offset)

    # Fallback: compute SDF from scratch when no cache is provided
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
