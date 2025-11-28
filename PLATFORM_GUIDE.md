# Face Detection & Recognition - Multi-Platform

Real-time face detection, recognition, mask detection, and object detection application with support for Windows and Linux.

## Features

- ✅ **Face Detection & Recognition** - Recognize known faces from a directory
- ✅ **Mask Detection** - Detect if person is wearing a mask
- ✅ **Object Detection** - YOLOv8-powered detection of 80+ object classes
- ✅ **Multi-Source Support** - Local camera, video files, IP cameras, RTSP streams
- ✅ **Cross-Platform** - Separate optimized versions for Windows and Linux
- ✅ **Live Display** - Real-time visualization with bounding boxes
- ✅ **Logging** - All detections saved to CSV file
- ✅ **Headless Mode** - Process without display (WSL, remote systems, etc.)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### 1. Clone/Download the Repository
```bash
cd Face_Detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Install YOLOv8 for Object Detection
```bash
pip install ultralytics
```

## Usage

### Windows Users
Use the Windows-optimized version with Tkinter display:

```bash
# Basic usage - webcam
python face_detection_windows.py --camera 0

# IP camera (e.g., IP Webcam app)
python face_detection_windows.py --video http://192.168.1.100:8080/video --display

# Video file
python face_detection_windows.py --video my_video.mp4 --display

# Without object detection (faster)
python face_detection_windows.py --camera 0 --no-object-detection

# Test camera before running
python face_detection_windows.py --test-camera
```

### Linux Users
Use the Linux-optimized version with OpenCV X11 display:

```bash
# Basic usage - webcam
python face_detection_linux.py --camera 0

# IP camera
python face_detection_linux.py --video http://192.168.1.100:8080/video

# Headless mode (no display)
python face_detection_linux.py --camera 0 --headless

# WSL with display
python face_detection_linux.py --camera 0 --display
```

## Setting Up Known Faces

1. Create a `known_faces` directory (created automatically on first run)
2. Add images of people you want to recognize
3. Name files as: `person_name.jpg` or `person_name.png`
4. The app will load and recognize these faces

Example structure:
```
known_faces/
├── john.jpg
├── sarah.png
└── mike.jpg
```

## Mobile Camera Setup (IP Webcam)

### Android
1. Install "IP Webcam" from Google Play Store
2. Open the app and note the IP address (e.g., `192.168.1.100`)
3. Run:
```bash
python face_detection_windows.py --video http://192.168.1.100:8080/video --display
```

### iPhone
Use apps like "Codeshot" or "Reincubate Codeshot"

## Command Line Options

### Common Options
```
--camera CAMERA_INDEX       Camera index to use (default: 0 for Linux, 1 for Windows)
--video VIDEO_PATH          Video file or stream URL
--known-faces DIR           Directory with known face images (default: known_faces/)
--log-file FILE             CSV file to log detections (default: detections.csv)
--test-camera               Test camera/video source and exit
--no-object-detection       Disable YOLOv8 (faster processing)
```

### Windows Only
```
--display                   Force display window (Tkinter)
--headless                  Disable display
```

### Linux Only
```
--display                   Force display window (OpenCV X11)
--headless                  Disable display (useful for WSL)
```

## Output

### Display Colors
- **Red box** = Unknown face detected
- **Green box** = Known face detected
- **Cyan box** = Object detected (cars, people, animals, etc.)

### CSV Log File
Detections are saved to `detections.csv`:
```
timestamp,person_name,mask_status,confidence
2025-11-28 14:30:45.123456,john,No Mask,0.95
2025-11-28 14:30:46.234567,Unknown,No Mask,0.0
```

## Performance Tips

1. **Disable Object Detection** if not needed:
   ```bash
   python face_detection_windows.py --camera 0 --no-object-detection
   ```

2. **Use Headless Mode** for faster processing:
   ```bash
   python face_detection_windows.py --video http://ip:8080/video --headless
   ```

3. **Resize Video Input** if too large
   ```bash
   python face_detection_windows.py --camera 0 --display
   ```

## Troubleshooting

### No Display on Windows
- Use `--display` flag explicitly
- Install Pillow: `pip install Pillow`
- Install Tkinter: It comes with Python on Windows

### No Display on Linux (WSL)
- Use `--headless` flag
- Or set up X11 forwarding for WSL

### Camera Not Found
- Test with: `python face_detection_windows.py --test-camera`
- Try different camera indices: 0, 1, 2, etc.
- Check if camera is in use by another application

### IP Camera Connection Failed
- Check IP address is correct
- Ensure device is on same network
- Test URL: Open in browser (for http://)
- Use `--test-camera` to validate URL

### Slow Performance
- Use `--no-object-detection` flag
- Use `--headless` mode
- Reduce input video resolution

## System Requirements

### Minimum
- CPU: Dual-core 2GHz
- RAM: 4GB
- GPU: Optional (for faster processing)

### Recommended
- CPU: Quad-core 3GHz
- RAM: 8GB
- GPU: NVIDIA CUDA (for YOLOv8 acceleration)

## Files

- `face_detection_windows.py` - Windows-optimized version
- `face_detection_linux.py` - Linux-optimized version
- `face_detection.py` - Original unified version (cross-platform)
- `detection_logger.py` - CSV logging module
- `requirements.txt` - Python dependencies
- `known_faces/` - Directory for known face images
- `detections.csv` - Detection log file

## License

See LICENSE file for details.

## Contributing

Feel free to contribute improvements!

## Support

For issues or questions, please open an issue on GitHub.
