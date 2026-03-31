import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Load the Data
# ==========================================
# IMPORTANT: Change this path to point to a specific .npy file in your folder!
file_path = r'C:\Users\lenovo\Downloads\prepareddataset\vehicle.car\3b36a446ccc34f53b22b4172d1938d2d.npy'

# Load the array into memory. It will have the shape (1024, 6)
object_data = np.load(file_path)
print(f"Loaded data shape: {object_data.shape}")

# Extract just the X, Y, and Z coordinates (the first 3 columns)
x = object_data[:, 0]
y = object_data[:, 1]
z = object_data[:, 2]

# (Optional) You can also see the velocity we attached!
vx, vy, vz = object_data[0, 3], object_data[0, 4], object_data[0, 5]
print(f"Object Velocity: [{vx:.2f}, {vy:.2f}, {vz:.2f}] m/s")

# ==========================================
# 2. Plot the 3D Point Cloud
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the points. 
# 'c=z' colors the points based on their height.
# 's=5' makes the points a bit larger and easier to see.
ax.scatter(x, y, z, c=z, cmap='plasma', s=5)

# Set the axis labels
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_zlabel('Z (meters)')
ax.set_title(f'PointNet Input Visualization\n1024 Points')

# Make the axes have an equal aspect ratio so the car doesn't look squished
# (Matplotlib requires a little math to make 3D boxes perfectly square)
max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
mid_x = (x.max()+x.min()) * 0.5
mid_y = (y.max()+y.min()) * 0.5
mid_z = (z.max()+z.min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Show the interactive plot!
plt.show()