import os
import numpy as np
from nuscenes.nuscenes import NuScenes

# 1. Initialize the NuScenes object
# UPDATE THESE VARIABLES:
# Change 'v1.0-mini' to 'v1.0-trainval' if you are using the full dataset.
# Change the dataroot to the path where your merged nuScenes folder lives.
DATASET_VERSION = 'v1.0-trainval' 
DATASET_ROOT = 'C:\\Users\\lenovo\\Downloads\\datasetlidar'  # e.g., '/home/user/nuscenes/v1.0-trainval'

print("Loading nuScenes metadata... (this might take a minute for the full dataset)")
nusc = NuScenes(version=DATASET_VERSION, dataroot=DATASET_ROOT, verbose=False)

# 2. Grab the very first sample (a single snapshot in time) from the dataset
first_scene = nusc.scene[0]
first_sample_token = first_scene['first_sample_token']
my_sample = nusc.get('sample', first_sample_token)

# 3. Get the token specifically for the Top LiDAR sensor
lidar_token = my_sample['data']['LIDAR_TOP']

# 4. Fetch the file paths for BOTH the raw point cloud and the lidarseg labels
# Get LiDAR path
lidar_metadata = nusc.get('sample_data', lidar_token)
pcl_path = os.path.join(nusc.dataroot, lidar_metadata['filename'])

# Get Lidarseg path (This will fail if the lidarseg.json isn't merged correctly!)
try:
    lidarseg_metadata = nusc.get('lidarseg', lidar_token)
    label_path = os.path.join(nusc.dataroot, lidarseg_metadata['filename'])
except KeyError:
    print("\n[ERROR] Could not find lidarseg labels. Make sure lidarseg.json is in your v1.0-* folder!")
    exit()

# 5. Load the raw binary data into NumPy arrays (What PointNet needs!)
# NuScenes LiDAR .bin files are saved as float32 with 5 channels: x, y, z, intensity, ring_index
points = np.fromfile(pcl_path, dtype=np.float32).reshape(-1, 5)

# NuScenes Lidarseg .bin files are saved as uint8 (one class label per point)
labels = np.fromfile(label_path, dtype=np.uint8)

# 6. Verify they match
print("\n--- Verification Results ---")
print(f"Point Cloud Array Shape: {points.shape}  -> (N points, 5 features)")
print(f"Labels Array Shape:      {labels.shape}      -> (N points, 1 label)")

if points.shape[0] == labels.shape[0]:
    print("\n✅ SUCCESS! Every 3D point has exactly one label.")
    print("Your dataset is merged correctly and ready for PointNet.")
else:
    print("\n❌ ERROR: Mismatch in points and labels. Something went wrong.")