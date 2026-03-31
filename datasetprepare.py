import os
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import points_in_box

# ==========================================
# 1. CONFIGURATION (Update these paths!)
# ==========================================
DATASET_VERSION = 'v1.0-trainval'  # You mentioned you are using the full dataset metadata
DATASET_ROOT = r'C:\Users\lenovo\Downloads\datasetlidar' # Your dataset folder
SAVE_ROOT = r'C:\Users\lenovo\Downloads\processed_nuscenes'  # Where to save the .npy files
NUM_POINTS = 1024 # Standard PointNet input size

print("Initializing nuScenes... (This will take a minute for the full dataset)")
nusc = NuScenes(version=DATASET_VERSION, dataroot=DATASET_ROOT, verbose=False)

# Ensure the main save folder exists
os.makedirs(SAVE_ROOT, exist_ok=True)

# Counters to keep track of progress
total_objects_saved = 0
skipped_empty_boxes = 0
missing_files_skipped = 0

print(f"\nStarting extraction for {len(nusc.scene)} scenes...")

# ==========================================
# 2. MAIN LOOP
# ==========================================
# Loop through every scene in the dataset
for scene_idx, scene in enumerate(nusc.scene):
    print(f"Processing Scene {scene_idx + 1} / {len(nusc.scene)}...")
    
    # Get the very first frame (sample) of this scene
    current_sample_token = scene['first_sample_token']
    
    # Loop through every frame in this scene until the end
    while current_sample_token != '':
        sample = nusc.get('sample', current_sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        
        # Get the point cloud path and all bounding boxes aligned to the LiDAR
        pcl_path, boxes, _ = nusc.get_sample_data(lidar_token)
        
        # --- BULLETPROOF TRY/EXCEPT ---
        try:
            # Try to load the raw 360-degree point cloud
            pc = LidarPointCloud.from_file(pcl_path)
        except FileNotFoundError:
            # We don't have this file (probably in Part 2-10). Skip to the next frame.
            missing_files_skipped += 1
            current_sample_token = sample['next']
            continue 
        # ------------------------------
        
        # Loop through every object (bounding box) in this frame
        for box in boxes:
            class_name = box.name
            
            # Find which points are inside this specific box
            mask = points_in_box(box, pc.points[:3, :])
            
            # Extract those points and transpose to shape (N, 3) for x, y, z
            object_points = pc.points[:3, mask].T 
            num_pts = object_points.shape[0]
            
            # If the box has fewer than 5 LiDAR points, it's too sparse to learn from. Skip it!
            if num_pts < 5:
                skipped_empty_boxes += 1
                continue
                
            # --- STANDARDIZE POINT COUNT ---
            if num_pts >= NUM_POINTS:
                # Downsample: Randomly pick exactly 1024 unique points
                choice = np.random.choice(num_pts, NUM_POINTS, replace=False)
            else:
                # Upsample: Randomly duplicate points until we reach 1024
                choice = np.random.choice(num_pts, NUM_POINTS, replace=True)
                
            standardized_points = object_points[choice, :] # Shape is now exactly (1024, 3)
            
            # --- ATTACH VELOCITY ---
            velocity = nusc.box_velocity(box.token)
            
            # Handle NaN values (if velocity can't be calculated, set it to 0.0)
            if np.any(np.isnan(velocity)):
                velocity = np.array([0.0, 0.0, 0.0])
                
            # Create a (1024, 3) array filled with the velocity
            velocity_array = np.tile(velocity, (NUM_POINTS, 1))
            
            # Glue coordinates and velocity together. Shape becomes (1024, 6)
            final_data = np.hstack((standardized_points, velocity_array)).astype(np.float32)
            
            # --- SAVE TO DISK ---
            # Create a folder for this specific class if it doesn't exist yet
            class_dir = os.path.join(SAVE_ROOT, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Name the file using the unique box token so we never overwrite files
            save_path = os.path.join(class_dir, f"{box.token}.npy")
            np.save(save_path, final_data)
            
            total_objects_saved += 1
            
        # Move to the next frame in the scene
        current_sample_token = sample['next']

print("\n=== EXTRACTION COMPLETE ===")
print(f"Successfully saved {total_objects_saved} objects.")
print(f"Skipped {skipped_empty_boxes} objects because they had too few points.")
print(f"Skipped {missing_files_skipped} missing LiDAR files.")