import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(PointNetClassifier, self).__init__()
        
        # ==========================================
        # 1. THE FEATURE EXTRACTOR (Shared MLPs)
        # ==========================================
        # We use 1D Convolutions with a kernel size of 1. 
        # This is PyTorch's trick to apply the exact same math to every single point independently.
        
        # INPUT: 6 channels (x, y, z, vx, vy, vz) -> OUTPUT: 64 features per point
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64) # Normalizes the data to help it learn faster
        
        # INPUT: 64 features -> OUTPUT: 128 features
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        
        # INPUT: 128 features -> OUTPUT: 1024 features
        # By this layer, every single point has been expanded into a 1024-dimensional feature vector!
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # ==========================================
        # 2. THE CLASSIFICATION HEAD (Fully Connected Layers)
        # ==========================================
        # After we mash all the points together, we pass the result through standard linear layers
        # to shrink it down to our 4 final classes (Person, Animal, Vehicle, Static).
        
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(p=0.3) # Randomly turns off neurons to prevent overfitting
        
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(p=0.3)
        
        # FINAL OUTPUT: 256 -> 4 classes
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        The 'forward' function dictates exactly how the data flows through the network.
        Input 'x' shape from our DataLoader: [Batch_Size, 6, 1024]
        """
        # Step 1: Extract features for every point individually
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x))) 
        # Shape is now: [Batch_Size, 1024_features, 1024_points]
        
        # Step 2: THE MAGIC POINTNET STEP (Global Max Pooling)
        # We find the maximum feature value across all 1024 points.
        # This destroys the "point" dimension, leaving us with one global signature for the whole object.
        x = torch.max(x, 2, keepdim=False)[0] 
        # Shape is now: [Batch_Size, 1024_features]
        
        # Step 3: Classify that global signature
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.fc3(x) 
        # Final Shape: [Batch_Size, 4] -> These are the raw prediction scores for our 4 classes!
        
        return x

