\# Advanced LiDAR-Based Surveillance System – Academic Project 2026



This academic project presents an advanced surveillance pipeline based on FMCW LiDAR technology. The system addresses key limitations of traditional video-based surveillance by using 3D point cloud data and Doppler radial velocity information for object classification, motion tracking, and real-time security alerting.



\## Project Overview



The system uses a PointNet-based deep learning architecture to classify 3D LiDAR point clouds into semantic categories such as:



\- Person

\- Vehicle

\- Obstacle



It also exploits Doppler radial velocity data from FMCW LiDAR sensors to improve object motion estimation, trajectory tracking, and multi-object surveillance analysis.



\## Main Features



\- FMCW LiDAR-based 3D surveillance pipeline

\- PointNet model for point cloud classification

\- Real-time classification of objects into semantic categories

\- Use of Doppler radial velocity for motion tracking

\- Multi-object trajectory analysis

\- Automated alerting module

\- Structured incident report generation

\- Geolocated security alerts



\## Technology Stack



\- Python

\- PyTorch

\- CNN / Deep Learning

\- PointNet

\- 3D Point Cloud Processing

\- Computer Vision

\- FMCW LiDAR

\- NumPy

\- Matplotlib



\## Project Architecture



```txt

.

├── datasetprepare.py      # Dataset preparation from LiDAR point clouds

├── datasetloader.py       # PyTorch dataset and dataloader

├── model.py               # PointNet classification model

├── train.py               # Model training script

├── evaluation.py          # Model evaluation script

├── visualizeobj.py        # 3D point cloud visualization

└── README.md

