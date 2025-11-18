# Intelligent_Autonomous_Driving




## Project Overview
Computer vision system for autonomous driving using stereo camera calibration, depth estimation, and 3D object detection.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Conda or pip
- CUDA 11.0+ (optional, for GPU acceleration)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Intelligent_Autonomous_Driving
   ```

2. **Activate Conda environment:**
   ```bash
   conda activate pfas
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import cv2, torch, ultralytics; print('All dependencies installed successfully!')"
   ```
5. **Download a pretrained YOLO detection Model**
https://docs.ultralytics.com/tasks/detect/#models

6. **Download stereo Image sequences uploaded in exam and place them in the project**
### Running the Project

- **Main script:** `python FinalProject.py`
- **Data location:** Place stereo image pairs in `Rectified Images/seq_*/left/` and `seq_*/right/` directories
- **Calibration data:** Place in `Rectified Images/calib_cam_to_cam.txt`
  ** Pretrained Model: ** Place in project root next to FinalProject.py
  ** Make sure to go throug FinalProject.py and verify all paths are correct for your machine**



## Tasks

## Timeline
- **Module Development Deadline:** 24 November 2025
- **Integration & Demo Week:** 25 November - 29 November 2025
- **Final Report Due:** 1 December 2025

### 1. Camera Calibration - Benedicte
- [ ] Calibrate camera using provided images
- [ ] Rectify test images
- [ ] Compare with provided calibration

### 2. Depth Estimation - Lars
- [ ] Preprocess images (downsampling, blur, etc.)
- [ ] Create depth map from stereo pairs

### 3. Object Detection & Tracking - TBD
- [ ] Detect objects in 2D
- [ ] Track pose (position & rotation) in 3D
- [ ] Classify: Bicycle, Person, Car

---

## Integration (25-29 Nov)
- [ ] Combine all modules
- [ ] Create demo video
- [ ] Finish report


