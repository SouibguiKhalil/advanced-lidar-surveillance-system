import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

class NuScenesPointNetDataset(Dataset):
    def __init__(self, data_root):
        """
        Reads the processed .npy files and maps them to 4 custom classes:
        0: Person, 1: Animal, 2: Vehicle, 3: Static Obstacle
        """
        self.data_root = data_root
        self.file_paths = []
        self.labels = []
        
        print("Scanning dataset folders...")
        
        # Loop through all the class folders we generated earlier
        for folder_name in os.listdir(data_root):
            folder_path = os.path.join(data_root, folder_name)
            if not os.path.isdir(folder_path):
                continue
            
            # --- MAP NUSCENES CLASSES TO YOUR 4 CUSTOM CLASSES ---
            if folder_name.startswith('human.'):
                label = 0  # Person
            elif folder_name.startswith('animal'):
                label = 1  # Animal
            elif folder_name.startswith('vehicle.'):
                label = 2  # Vehicle
            elif folder_name.startswith('movable_object.') or folder_name.startswith('static_object.'):
                label = 3  # Static Obstacle
            else:
                continue # Skip anything else we don't care about
                
            # Grab every .npy file inside this mapped folder
            files_in_folder = os.listdir(folder_path)
            for file_name in files_in_folder:
                if file_name.endswith('.npy'):
                    self.file_paths.append(os.path.join(folder_path, file_name))
                    self.labels.append(label)
                    
        print(f"Successfully loaded {len(self.file_paths)} total objects!")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 1. Load the raw numpy array from disk. Shape: (1024, 6)
        pc_np = np.load(self.file_paths[idx])
        
        # 2. Convert to a PyTorch float tensor
        pc_tensor = torch.tensor(pc_np, dtype=torch.float32)
        
        # 3. CRITICAL POINTNET FIX: Transpose the matrix
        pc_tensor = pc_tensor.transpose(0, 1)
        
        # 4. Get the target label (0, 1, 2, or 3)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return pc_tensor, label_tensor

# ==========================================
# DATALOADER INITIALIZATION 
# ==========================================
# Point this to the folder where you saved all the .npy files!
DATA_DIR = r'C:\Users\lenovo\Downloads\prepareddataset'

# 1. Initialize the full dataset
full_dataset = NuScenesPointNetDataset(data_root=DATA_DIR)

# 2. Calculate the split sizes (80% Train, 20% Test)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

# 3. Randomly divide the dataset (with a fixed seed for reproducibility)
generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)

# 4. Create the DataLoaders
# train.py will use train_dataloader
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# evaluate.py will use test_dataloader
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# ==========================================
# TEST THE DATALOADER (Only runs if you execute this file directly)
# ==========================================
if __name__ == "__main__":
    print(f"\n=== Dataset Split Info ===")
    print(f"Total objects: {len(full_dataset)}")
    print(f"Training objects (80%): {len(train_dataset)}")
    print(f"Testing objects (20%): {len(test_dataset)}")
    
    # Grab one batch to test it
    for points, labels in train_dataloader:
        print("\n--- Train Batch Test ---")
        print(f"Points shape: {points.shape}") # Should be [32, 6, 1024]
        print(f"Labels shape: {labels.shape}") # Should be [32]
        print(f"Sample labels in batch: {labels[:5].tolist()}")
        break # Just testing the first batch, so we stop here