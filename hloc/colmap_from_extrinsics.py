import argparse
import sqlite3
import numpy as np
from pathlib import Path
from collections import defaultdict
import pycolmap
from typing import Dict, List, Optional, Any

from . import logger
from .colmap_from_nvm import quaternion_to_rotation_matrix
from .utils.read_write_model import (
    Camera,
    Image,
    write_model,
    CAMERA_MODEL_NAMES,
)
from .reconstruction import create_empty_db, import_images, get_image_ids

# def recover_database_ids(database_path):
#     """
#     Recovers the mapping from image name to database image_id.
#     Crucial to ensure the model aligns with the feature database.
#     """
#     name2id = {}
#     db = sqlite3.connect(str(database_path))
#     ret = db.execute("SELECT name, image_id FROM images;")
#     for name, image_id in ret:
#         name2id[name] = image_id
#     db.close()
#     logger.info(f"Found {len(name2id)} images in database.")
#     return name2id

def read_intrinsics(intrinsics_path):
    """
    Reads intrinsics.txt. 
    Format per line: CAMERA_ID MODEL WIDTH HEIGHT PARAMS...
    Example: 1 PINHOLE 1920 1080 1000 1000 960 540
    """
    cameras = {}
    with open(intrinsics_path, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
            
        data = line.split()
        cam_id = int(data[0])
        model_name = data[1]
        width = int(data[2])
        height = int(data[3])
        params = [float(p) for p in data[4:]]
        
        model = CAMERA_MODEL_NAMES[model_name]
        assert len(params) == model.num_params, f"Params mismatch for {model_name}"
        
        cameras[cam_id] = Camera(
            id=cam_id,
            model=model_name,
            width=width,
            height=height,
            params=np.array(params),
        )
    logger.info(f"Loaded {len(cameras)} cameras.")
    
    return cameras


# def generate_camera(intrinsics, model_name, width, height, camera_id = None):
#     """
#     Generate a Camera object given intrinsics and model.
#     intrinsics: list or np.array of parameters
#     model_name: string, e.g. 'PINHOLE'
#     width: int
#     height: int
#     camera_id: int or None
#     """
#     model = CAMERA_MODEL_NAMES[model_name]
#     assert len(intrinsics) == model.num_params, f"Params mismatch for {model_name}"
#     return Camera(
#         id=camera_id if camera_id is not None else 1,
#         model=model_name,
#         width=width,
#         height=height,
#         params=np.array(intrinsics),
#     )


def read_poses(poses_path, name2id, cameras, format='COLMAP'):
    logger.info(f"Reading poses from {poses_path} (Format: {format})...")
    if format == 'COLMAP':
        return read_poses_colmap(poses_path, name2id, cameras)
    elif format == 'TUM':
        return read_poses_tum(poses_path, name2id, cameras)
    else:
        raise ValueError(f"Unknown pose format: {format}")

    
def read_poses_tum(poses_path: Path, name2id: Dict[str, int], cameras: Dict[int, Camera]):
    """
    Reads poses.txt in TUM format.
    Standard TUM: timestamp tx ty tz qx qy qz qw
    Assumption: All images use the FIRST camera found in 'cameras' dict, 
    since TUM format doesn't support camera_ids.
    """
    images = {}
    # Default to the first camera ID provided in intrinsics
    default_camera_id = list(cameras.keys())[0]
    
    with open(poses_path, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
            
        data = line.split()
        name = data[0].strip().split("/")[-1] # timestamp or filename
        
        # Check if this image exists in our database (imported from folder)
        if name not in name2id:
            # Try appending extensions if missing, or generic matching?
            # For now, strict match.
            continue
            
        # Parse TUM: tx ty tz qx qy qz qw
        tx, ty, tz = float(data[1]), float(data[2]), float(data[3])
        qx, qy, qz, qw = float(data[4]), float(data[5]), float(data[6]), float(data[7])
        
        # TUM is usually [qx, qy, qz, qw] (scipy default). 
        # COLMAP/Our Helpers usually expect [qw, qx, qy, qz].
        # Let's align to [qw, qx, qy, qz] for the conversion function.
        qvec_tum_wfirst = np.array([qw, qx, qy, qz])
        tvec_tum = np.array([tx, ty, tz])
        
        # Convert C2W (TUM) -> W2C (COLMAP)
        qvec_colmap, tvec_colmap = convert_from_tum_to_colmap(qvec_tum_wfirst, tvec_tum)
        
        image_id = name2id[name]
        
        images[image_id] = Image(
            id=image_id,
            qvec=qvec_colmap,
            tvec=tvec_colmap,
            camera_id=default_camera_id,
            name=name,
            xys=np.zeros((0, 2), float),
            point3D_ids=np.full(0, -1, int),
        )
        
    logger.info(f"Loaded {len(images)} images from TUM poses.")
    return images

def read_poses_colmap(poses_path, name2id, cameras):
    """
    Reads poses.txt in COLMAP format (World-to-Camera).
    Format per line: IMAGE_NAME CAMERA_ID QW QX QY QZ TX TY TZ
    """
    images = {}
    with open(poses_path, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
            
        data = line.split()
        name = data[0]
        
        if name not in name2id:
            logger.warning(f"Image {name} in poses.txt not found in database. Skipping.")
            continue
            
        camera_id = int(data[1])
        assert camera_id in cameras, f"Camera ID {camera_id} not found in intrinsics."
        
        # Parse Pose (World-to-Camera)
        qvec = np.array([float(x) for x in data[2:6]])
        tvec = np.array([float(x) for x in data[6:9]])
        
        image_id = name2id[name]
        
        # Create Image object (points3D_ids are empty/dummy for now)
        images[image_id] = Image(
            id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=name,
            xys=np.zeros((0, 2), float),
            point3D_ids=np.full(0, -1, int),
        )
        
    logger.info(f"Loaded {len(images)} images with poses.")
    return images
    

def convert_from_tum_to_colmap(qvec_tum, tvec_tum):
    """
    Converts TUM format (Camera-to-World) to COLMAP format (World-to-Camera).
    Input:
      qvec_tum: [w, x, y, z] (Camera-to-World orientation)
      tvec_tum: [tx, ty, tz] (Camera-to-World translation)
    Output:
      qvec_colmap, tvec_colmap (World-to-Camera)
    """
    # 1. Ensure input quaternion is normalized (Important!)
    qvec_tum = qvec_tum / np.linalg.norm(qvec_tum)

    # 2. Invert Rotation (Conjugate is the most accurate method)
    # q_inv = [w, -x, -y, -z]
    qvec_colmap = np.array([qvec_tum[0], -qvec_tum[1], -qvec_tum[2], -qvec_tum[3]])

    # 3. Invert Translation
    # Formula: t_w2c = -R_w2c * t_c2w
    # We need the rotation matrix of the *inverse* (World-to-Camera) rotation
    R_w2c = quaternion_to_rotation_matrix(qvec_colmap)
    tvec_colmap = -R_w2c @ tvec_tum
    
    return qvec_colmap, tvec_colmap

def main(
    output_dir: Path, 
    image_dir: Path, 
    poses: Path, 
    intrinsics: Path, 
    camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.AUTO,
    format: str = 'COLMAP',
    image_list: Optional[List[str]] = None,
    image_options: Optional[Dict[str, Any]] = None,
) -> None:
    
    assert poses.exists(), f"Poses file not found: {poses}"
    assert intrinsics.exists(), f"Intrinsics file not found: {intrinsics}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    database = output_dir / "database.db"
    
    # 1. Initialize Database & Import Images
    # This assigns Image IDs and handles filenames
    create_empty_db(database)
    import_images(image_dir, database, camera_mode, image_list, image_options)
    
    # 2. Get consistent IDs from Database
    # We need to know which ID COLMAP assigned to 'img1.jpg'
    name2id = get_image_ids(database)
    
    # 3. Read Intrinsics (Cameras)
    cameras = read_intrinsics(intrinsics)
    
    # 4. Read Extrinsics (Images/Poses)
    # This matches the poses to the IDs from step 2
    images = read_poses(poses, name2id, cameras, format=format)
    
    # 5. Create Empty Points3D
    points3D = {} 

    # 6. Write Model
    # We write the model to the same output directory
    logger.info("Writing reference model...")
    write_model(cameras, images, points3D, str(output_dir), ext=".bin")
    logger.info(f"Reference model (Extrinsics+Intrinsics) saved to {output_dir}")
    logger.info("You can now run hloc.triangulation using this directory as 'reference_model'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a COLMAP reference model from extrinsics/intrinsics text files.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Output directory for model and database")
    parser.add_argument("--image_dir", required=True, type=Path, help="Directory containing images")
    parser.add_argument("--poses", required=True, type=Path, help="Path to poses.txt")
    parser.add_argument("--intrinsics", required=True, type=Path, help="Path to intrinsics.txt")
    parser.add_argument("--format", type=str, default='COLMAP', choices=['COLMAP', 'TUM'], help="Format of the poses file")
    
    # Optional args for pycolmap options
    parser.add_argument("--camera_mode", type=str, default="AUTO", choices=[m for m in pycolmap.CameraMode.__members__])
    
    args = parser.parse_args()
    
    # Convert camera_mode string to Enum
    if args.camera_mode == "SINGLE":
        camera_mode = pycolmap.CameraMode.SINGLE
    else:
        camera_mode = pycolmap.CameraMode.AUTO
    
    main(
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        poses=args.poses,
        intrinsics=args.intrinsics,
        camera_mode=camera_mode,
        format=args.format
    )