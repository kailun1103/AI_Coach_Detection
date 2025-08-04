# AI Coach Detection

A real-time tennis motion analysis and feedback system that performs 3D pose estimation using linear triangulation and visualizes results in a web-based interface.


## 🎥 Demo

Watch the demo walkthrough on YouTube:

[![VIDEO](https://img.youtube.com/vi/VvKFSDbbvzY/0.jpg)](https://www.youtube.com/watch?v=VvKFSDbbvzY)


---

## ✨ Features

- **3D Pose Estimation**  
  Implements linear triangulation to reconstruct 3D joint positions from multi-view or calibrated camera setups.  
- **Real-Time Feedback**  
  Provides instant motion feedback on swing technique, posture, and trajectory deviations.  
- **Web Visualization**  
  Renders 3D skeleton and motion trails in a browser using Three.js.  
- **Performance Optimization**  
  Parallelized Python backend and GPU-accelerated CUDA modules for reducing computation time from 30s per frame to ~8s.  
- **Customizable Parameters**  
  Configure camera calibration, skeleton model, and feedback thresholds via JSON settings.

---

## 🛠 Tech Stack

- **Backend:** Python 3.8+, OpenCV, NumPy  
- **Pose Computation:** Custom linear triangulation module (CUDA-enabled)  
- **Visualization:** Three.js, HTML5, JavaScript  
- **Server:** Flask (or FastAPI) for streaming pose data  
- **GPU Acceleration:** CUDA Toolkit  

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**  
- **Node.js & npm**  
- **CUDA Toolkit** (for GPU support)  
- **Calibrated Cameras** or dataset with known intrinsics/extrinsics  

### Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/kailun1103/AI_Coach_Detection.git
   cd AI_Coach_Detection
   ```

2. **Backend Setup**  
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Frontend Setup**  
   ```bash
   cd ../frontend
   npm install
   npm run build
   ```

4. **Configure Settings**  
   - Copy `config_template.json` to `config.json` in both `backend` and `frontend` folders.  
   - Edit camera parameters, model paths, and feedback thresholds.

5. **Run the Application**  
   - Start backend server:
     ```bash
     cd backend
     python server.py
     ```
   - Serve frontend:
     ```bash
     cd frontend
     npm run start
     ```
   - Open `http://localhost:3000` in your browser.

---

## 📂 Project Structure

```
AI_Coach_Detection/
├── backend/
│   ├── server.py
│   ├── pose_triangulation.py
│   ├── cuda_module/
│   ├── requirements.txt
│   └── config_template.json
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── config_template.json
├── assets/
│   └── demo.gif
└── README.md
```

---

## 🤝 Contributing

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/YourFeature`)  
3. Commit your changes (`git commit -m 'Add feature'`)  
4. Push to the branch (`git push origin feature/YourFeature`)  
5. Open a Pull Request  

Please adhere to the existing code style and include meaningful commit messages.

---

## 📄 License

This project is licensed under the MIT License:

```
MIT License

Copyright (c) 2025 Kellen Chang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
