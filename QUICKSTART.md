# ⚡ Quick Start Guide

Get up and running in 5 minutes!

## Step 1: Setup (1 minute)

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install YOLOv8 for object detection
pip install ultralytics

# Run setup script
python setup.py
```

## Step 2: Add Known Faces (Optional)

```
Create known_faces/ directory with images:
known_faces/
├── john.jpg
├── sarah.png
└── mike.jpg
```

## Step 3: Run Application

### 🪟 Windows Users
```bash
# Webcam
python face_detection_windows.py --camera 0 --display

# IP Camera (Android - IP Webcam app)
python face_detection_windows.py --video http://192.168.1.3:4747/video --display
```

### 🐧 Linux Users
```bash
# Webcam
python face_detection_linux.py --camera 0

# IP Camera
python face_detection_linux.py --video http://192.168.1.3:4747/video

# Headless (no display)
python face_detection_linux.py --camera 0 --headless
```

## Common Commands

```bash
# Test camera before running
python setup.py

# Run with different camera
python face_detection_windows.py --camera 1

# Faster processing (no object detection)
python face_detection_windows.py --camera 0 --no-object-detection

# Process video file
python face_detection_windows.py --video my_video.mp4 --display

# Check detections
# Open detections.csv file
```

## Mobile Camera Setup (Android)

1. **Install IP Webcam** from Google Play Store
2. **Open app** → Note the IP address shown (e.g., `http://192.168.1.3:4747`)
3. **Run this command:**
```bash
python face_detection_windows.py --video http://192.168.1.3:4747/video --display
```

## Output

- **Red Box** = Unknown face
- **Green Box** = Known face + mask status
- **Cyan Box** = Objects detected

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No camera found | Run `python setup.py` to test |
| Display not showing | Use `--display` flag or check Pillow installed |
| Slow performance | Add `--no-object-detection` flag |
| IP camera won't connect | Check IP address and network connection |

## Need Help?

- See **README.md** for full documentation
- See **PLATFORM_GUIDE.md** for detailed guides
- Run `python face_detection_windows.py --help` for all options

---

That's it! You're ready to go! 🎉
