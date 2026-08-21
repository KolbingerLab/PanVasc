import os
import glob
import numpy as np
import nibabel as nib
import pyvista as pv
from skimage.measure import marching_cubes
from scipy.spatial import cKDTree
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

seg_dir = '/path/to/segmentationfiles/stored/as/.nii.gz/'
out_dir = '/output/folder/'
os.makedirs(out_dir, exist_ok=True)

# Segmentation labels
VESSEL_LABELS = {
    "CA": 2,
    "SMA": 8,
    "Veins": 9,
}

TUMOR_LABEL = 10

# Rendering settings
max_distance_mm = 50.0 # Can change
smooth_iterations = 20
window_size = (2200, 1600)
off_screen = True

def patient_id_from_path(path):
    base = os.path.basename(path)
    if base.endswith('.nii.gz'):
        return base[:-7]
    elif base.endswith('.nii'):
        return base[:-4]
    return os.path.splitext(base)[0]

def mask_to_mesh(mask, affine, smooth_iterations=50):
    if np.count_nonzero(mask) < 10:
        return None

    # Extracting true voxel size from the nifti header
    voxel_sizes = tuple(np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).tolist())

    mask_padded = np.pad(mask.astype(np.uint8), pad_width=2, mode="constant")

    # Passing spacing so that marching cube generates physically correct geometry
    verts, faces, _, _ = marching_cubes(
        mask_padded,
        level=0.5,
        spacing=voxel_sizes
    )
    verts = verts - np.array(voxel_sizes) * 2  # for padding in mm

    faces_pv = np.hstack([
        np.full((faces.shape[0], 1), 3),
        faces
    ]).astype(np.int64)

    mesh = pv.PolyData(verts, faces_pv)
    mesh = mesh.triangulate().clean()

    try:
        mesh = mesh.smooth_taubin(
            n_iter=80,
            pass_band=0.05,
            normalize_coordinates=True
        )
    except Exception:
        mesh = mesh.smooth(n_iter=80, relaxation_factor=0.05)

    mesh = mesh.compute_normals(
        point_normals=True, cell_normals=False,
        auto_orient_normals=True, consistent_normals=True,
        inplace=False
    )
    return mesh

def add_distance_to_tumor(vessel_mesh, tumor_mesh, max_distance_mm=15.0):
    # densifying tumor surface
    tumor_dense = tumor_mesh.subdivide(2, subfilter="butterfly")
    tumor_tree = cKDTree(tumor_dense.points)
    
    distances, _ = tumor_tree.query(vessel_mesh.points, workers=-1)
    distances = np.clip(distances, 0, max_distance_mm)
    vessel_mesh["distance_to_tumor_mm"] = distances
    return vessel_mesh


def setup_plotter(window_size=(2200, 1600), off_screen=True):
    plotter = pv.Plotter(
        off_screen=off_screen,
        window_size=window_size
    )

    plotter.set_background("white")

    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception:
        pass

    try:
        plotter.enable_depth_peeling()
    except Exception:
        pass

    try:
        plotter.enable_ssao(radius=8.0, bias=0.01, kernel_size=128)
    except Exception:
        pass

    return plotter

distance_cmap = 'Spectral'

def render_single_vessel(patient_id, vessel_name, vessel_mesh, tumor_mesh,
                         out_png, max_distance_mm=15.0, window_size=(2200, 1600), 
                         off_screen=True):
    
    plotter = setup_plotter(
        window_size=window_size,
        off_screen=off_screen
    )

    # Tumor
    plotter.add_mesh(
        tumor_mesh,
        color="orangered",
        opacity=0.25,
        smooth_shading=True,
        specular=0.3,
        specular_power=30
    )

    # Vessel colored by distance to tumor
    plotter.add_mesh(
        vessel_mesh,
        scalars="distance_to_tumor_mm",
        cmap=distance_cmap,
        clim=[0, max_distance_mm],
        opacity=1.0,
        smooth_shading=True,
        specular=0.5,
        specular_power=40,
        scalar_bar_args={
            "title": "",
            "vertical": False,
            "position_x": 0.22,
            "position_y": 0.04,
            "width": 0.56,
            "height": 0.08,
            "title_font_size": 30,
            "label_font_size": 24,
            "color": "black",
        }
    )

    plotter.add_axes()
    plotter.camera_position = "xz"
    plotter.reset_camera()
    plotter.camera.zoom(1.00)

    plotter.screenshot(out_png)
    plotter.close()

def render_all_vessels(patient_id, vessel_meshes, tumor_mesh, 
                       out_png, max_distance_mm=15.0, window_size=(2200, 1600),
                       off_screen=True):
    
    plotter = setup_plotter(window_size=window_size, off_screen=off_screen)
   
    plotter.add_mesh(
        tumor_mesh,
        color="orangered",
        opacity=0.25,
        smooth_shading=True,
        specular=0.3,
        specular_power=30
    )
    
    for i, (vessel_name, vmesh) in enumerate(vessel_meshes.items()):
        show_bar = (i == 0)
        plotter.add_mesh(
            vmesh,
            scalars="distance_to_tumor_mm",
            cmap=distance_cmap,
            clim=[0, max_distance_mm],
            opacity=1.0,
            smooth_shading=True,
            specular=0.5,
            specular_power=40,
            scalar_bar_args={
                "title": "",
                "vertical": False,
                "position_x": 0.22,
                "position_y": 0.04,
                "width": 0.56,
                "height": 0.08,
                "title_font_size": 30,
                "label_font_size": 24,
                "color": "black",
            } if show_bar else None,
            show_scalar_bar=show_bar,
        )
    plotter.add_axes()
    plotter.camera_position = "xz"
    plotter.reset_camera()
    plotter.camera.zoom(1.00)
    plotter.screenshot(out_png)
    plotter.close()


def process_one_patient_tg(seg_path):
    patient_id = patient_id_from_path(seg_path)
    patient_out_dir = os.path.join(out_dir, patient_id)
    os.makedirs(patient_out_dir, exist_ok=True)
    print(f"\nProcessing patient: {patient_id}")
    nii = nib.load(seg_path)
    seg = np.asanyarray(nii.dataobj)
    seg = np.rint(seg).astype(np.int16)
    affine = nii.affine

    tumor_mask = seg == TUMOR_LABEL
    if np.count_nonzero(tumor_mask) == 0:
        print(f"  Skipping: tumor label {TUMOR_LABEL} not found.")
        return
    tumor_mesh = mask_to_mesh(
        tumor_mask,
        affine,
        smooth_iterations=smooth_iterations
    )
    if tumor_mesh is None:
        print("  Skipping: tumor mesh failed.")
        return
    saved_images = []
    vessel_meshes = {}

    for vessel_name, label in VESSEL_LABELS.items():
        vessel_mask = seg == label
        if np.count_nonzero(vessel_mask) == 0:
            print(f"  Missing {vessel_name}, label {label}")
            continue
        print(f"  Rendering {vessel_name}, label {label}")
        vessel_mesh = mask_to_mesh(
            vessel_mask,
            affine,
            smooth_iterations=smooth_iterations
        )
        if vessel_mesh is None:
            print(f"  Failed mesh for {vessel_name}")
            continue
        vessel_mesh = add_distance_to_tumor(
            vessel_mesh,
            tumor_mesh,
            max_distance_mm=max_distance_mm
        )
        vessel_meshes[vessel_name] = vessel_mesh    # ← store

        mesh_out = os.path.join(patient_out_dir, f"{vessel_name}_colored_mesh.vtp")
        vessel_mesh.save(mesh_out)

        png_out = os.path.join(patient_out_dir, f"{vessel_name}_distance_render.png")
        render_single_vessel(
            patient_id=patient_id,
            vessel_name=vessel_name,
            vessel_mesh=vessel_mesh,
            tumor_mesh=tumor_mesh,
            out_png=png_out,
            max_distance_mm=max_distance_mm,
            window_size=window_size,
            off_screen=off_screen
        )
        saved_images.append(png_out)
        print(f"    Saved image: {png_out}")
        print(f"    Saved mesh:  {mesh_out}")

    if vessel_meshes:
        combined_out = os.path.join(patient_out_dir, "all_vessels_distance_render.png")
        render_all_vessels(
            patient_id=patient_id,
            vessel_meshes=vessel_meshes,
            tumor_mesh=tumor_mesh,
            out_png=combined_out,
            max_distance_mm=max_distance_mm,
            window_size=window_size,
            off_screen=off_screen
        )
        saved_images.append(combined_out)
        print(f"    Saved combined image: {combined_out}")

    return saved_images

# Running
if __name__ == "__main__":
    seg_dir = "/path/to/segmentations"

    for img in sorted(os.listdir(seg_dir)):
        seg_path = os.path.join(seg_dir, img)
        process_one_patient_tg(seg_path)    
