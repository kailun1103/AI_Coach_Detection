# Tennis Swing Analysis System

## Overview

This project is a comprehensive tennis swing analysis system that uses computer vision, 3D trajectory reconstruction, and machine learning to provide feedback on tennis swings. The system captures video from two camera angles, processes the footage to track body keypoints and the tennis ball, reconstructs the 3D trajectory, and then provides automated feedback on the player's technique.

## Key Components

### 1. Camera Calibration & 3D Reconstruction

- Uses dual-camera setup (side view and 45° angle view)
- Implements binocular correction to calibrate cameras
- Reconstructs 3D trajectories from 2D keypoints using triangulation

### 2. Motion Tracking

- Utilizes YOLOv8 pose detection to track 17 key body points
- Employs custom tennis ball detection model
- Provides real-time visualization of trajectories

### 3. Data Processing Pipeline

- 2D trajectory extraction and smoothing
- Video synchronization for multi-angle analysis
- 3D trajectory calculation and filtering
- Automated swing range detection

### 4. Feedback System

- K-Nearest Neighbors (KNN) analysis to compare with reference swings
- AI-powered feedback using language models
- Provides specific frame ranges where issues appear
- Generates comprehensive improvement suggestions

### 5. Web Visualization

- Interactive 3D viewer for swing trajectories
- Control panel for playback and analysis options
- Color-coded feedback visualization
- Comparison tools for multiple swings

## Technical Implementation

- Backend: Python, FastAPI
- Computer Vision: OpenCV, YOLOv8, NumPy
- 3D Visualization: Three.js
- Machine Learning: scikit-learn, GPT models
- Video Processing: FFmpeg

## Usage

The system captures videos of a tennis player from two angles, processes them through the analysis pipeline, and generates feedback that can be viewed in the web interface. Players can record multiple swings, compare them, and track improvement over time.

## Future Directions

- Integration with mobile devices
- Real-time analysis capabilities
- Expanded database of reference swings
- Support for additional shot types and techniques

---

This system aims to democratize access to professional-level tennis coaching by providing detailed technical analysis and actionable feedback to players of all levels.