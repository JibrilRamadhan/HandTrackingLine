# 🖐️ Neon Finger Tracking AR

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks_Vision-orange.svg)
![Vercel](https://img.shields.io/badge/Deployed_on-Vercel-black.svg)

A high-performance, real-time Computer Vision application that tracks your fingers to create dynamic, glowing shapes and visual effects. Recreating popular TikTok AR trends using **Python (OpenCV + MediaPipe)** and a lightweight **Web AR** version that runs entirely in your browser.

---

## ✨ Features

- **👐 Multi-Hand Tracking**: Accurately tracks up to 2 hands simultaneously.
- **💠 Dynamic Shapes**: Connects your fingertips in real-time to form shapes:
  - `WEB` (Connects all 10 fingers continuously)
  - `LINE`, `TRIANGLE`, `RECTANGLE`, `DIAMOND`
- **🎨 Inner Fill Effects**: Automatically cycles beautiful internal rendering effects inside your shapes:
  - `BW` (High-contrast Black & White)
  - `HEATMAP` / `RAINBOW` (Vibrant Neon Glow)
  - `HALFTONE` (Retro dot-matrix style)
- **🏹 Interactive Gestures**: 
  - **Bow & Arrow:** Pinch your thumb and index finger on one hand, pull back from the other, and release to shoot glowing animated arrows!
- **📱 Cross-Platform**: Run it on your PC via Python, use your Android phone as a wireless webcam, or host the Web version on Vercel to run directly in a mobile browser.

---

## 🚀 1. Web Version (Recommended for Mobile)

The web version runs purely on client-side Javascript. No installation required. Perfect for testing on Android/iOS.

### 🌐 Live Demo
*(If you hosted this on Vercel, put your link here: `https://your-vercel-link.vercel.app`)*

### 💻 Run Locally
```bash
cd web
python -m http.server 8000
```
Then open `http://localhost:8000` in your browser.

---

## 🐍 2. Python Version (For PC)

The Python script offers maximum performance utilizing local CPU/GPU resources and OpenCV.

### 🛠️ Installation
1. Clone this repository:
```bash
git clone https://github.com/JibrilRamadhan/HandTrackingLine.git
cd HandTrackingLine
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
*(The script will automatically download the required MediaPipe `hand_landmarker.task` AI model on its first run).*

### 🎮 Usage

**Use Laptop Webcam (Default)**
```bash
python neon_finger_tracking.py
```

**Use HP/Smartphone Camera (via IP Webcam)**
1. Install **IP Webcam** from the Google Play Store on your Android phone.
2. Open the app, scroll down, and tap **"Start server"**.
3. Note the IP address shown on the screen (e.g., `http://192.168.1.5:8080`).
4. Run the script with the `--source` or `-s` flag:
```bash
python neon_finger_tracking.py -s http://192.168.1.5:8080/video
```
*(Tip: To reduce delay over WiFi, set the video resolution in the IP Webcam app to 640x480).*

### ⌨️ Python Controls
| Key | Action |
|-----|--------|
| `M` | Cycle through Shape Modes (Web, Line, Triangle, etc.) |
| `F` | Cycle through Fill Effects (BW, Heatmap, Halftone) |
| `T` | Toggle Fingertip Trails |
| `G` | Toggle Outer Neon Glow |
| `1-5` | Change Colors (White, Cyan, Magenta, Lime, Rainbow) |
| `C` | Clear Trails |
| `Q` or `ESC` | Quit |

---

## 🛠️ Built With
- **[MediaPipe Tasks Vision](https://developers.google.com/mediapipe)** - World-class ML hand tracking
- **[OpenCV](https://opencv.org/)** - Real-time computer vision in Python
- **[Vanilla JS & Canvas API](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API)** - For the Web AR rendering

---
*Created by [Jibril Ramadhan](https://github.com/JibrilRamadhan)*
