# 🎥 Face Detection & Recognition System

A powerful real-time face detection, recognition, and object detection application with support for multiple video sources and cross-platform compatibility.

## ✨ Key Features

- 👤 **Face Detection & Recognition** - Detect and recognize known faces in real-time
- 🎭 **Mask Detection** - Identify if person is wearing a face mask
- 🤖 **YOLOv8 Object Detection** - Detect 80+ object classes (cars, people, animals, etc.)
- 📹 **Multi-Source Support**:
  - Local USB cameras (any index)
  - IP cameras (IP Webcam app on Android)
  - RTSP streams
  - Video files
- 💻 **Cross-Platform**:
  - Windows (Tkinter display)
  - Linux (OpenCV X11 display)
  - WSL (headless mode)
- 🖼️ **Live Visualization** - Real-time display with bounding boxes and labels
- 📊 **CSV Logging** - All detections automatically logged with timestamp
- ⚡ **Headless Mode** - Process without display for servers/remote systems
- 🔧 **Easy Configuration** - Simple command-line interface

## 📋 Requirements

- **Python**: 3.8 or higher
- **OS**: Windows, Linux, or WSL
- **Camera**: USB webcam or IP camera (optional)

## 🚀 Quick Start

### 1. Installation

```bash
# Navigate to project
cd Face_Detection

# Install dependencies
pip install -r requirements.txt

# (Optional) Install YOLOv8 for object detection
pip install ultralytics
```

### 2. Add Known Faces

Create `known_faces/` directory with images:
```
known_faces/
├── john.jpg
├── sarah.png
└── mike.jpg
```

### 3. Run the Application

**Windows:**
```bash
python face_detection_windows.py --camera 0 --display
```

**Linux:**
```bash
python face_detection_linux.py --camera 0
```

**With IP Camera (Android):**
```bash
python face_detection_windows.py --video http://192.168.1.100:4747/video --display
```

## 🎮 Quick Examples

### Windows
```bash
# Webcam with display
python face_detection_windows.py --camera 0 --display

# IP Webcam from phone
python face_detection_windows.py --video http://192.168.1.3:4747/video --display

# Video file
python face_detection_windows.py --video video.mp4 --display

# Faster (no object detection)
python face_detection_windows.py --camera 0 --no-object-detection

# Test camera
python face_detection_windows.py --test-camera
```

### Linux
```bash
# Webcam
python face_detection_linux.py --camera 0

# IP camera
python face_detection_linux.py --video http://192.168.1.100:4747/video

# Headless mode
python face_detection_linux.py --camera 0 --headless
```

## 📱 Mobile Camera Setup

### Android - IP Webcam (Recommended)

1. Install **IP Webcam** from Google Play Store
2. Open app and note the IP address displayed (e.g., `http://192.168.1.3:4747`)
3. Run:
```bash
python face_detection_windows.py --video http://192.168.1.3:4747/video --display
```

### iPhone

Use apps like **Codeshot** or **Reincubate Codeshot**

## 🎨 Display Output

- **🔴 Red Box** = Unknown face
- **🟢 Green Box** = Known face (with name and mask status)
- **🔵 Cyan Box** = Objects detected (cars, people, animals, etc.)

## 📊 CSV Log Output

Detections saved to `detections.csv`:
```csv
timestamp,person_name,mask_status,confidence
2025-11-28 14:30:45.123456,john,No Mask,0.95
2025-11-28 14:30:46.234567,Unknown,No Mask,0.0
2025-11-28 14:30:47.345678,sarah,Mask,0.98
```

## 📖 Documentation

For detailed platform-specific guides, see:
- 📄 **PLATFORM_GUIDE.md** - Windows & Linux specific instructions
- 🔧 **Command-line options** - Complete list of arguments

## ⚙️ Command-Line Options

```
--camera INDEX              Camera index (0, 1, 2, etc.)
--video URL_OR_PATH         Video file, IP camera URL, or RTSP stream
--known-faces DIR           Directory with face images (default: known_faces/)
--log-file FILE             CSV output file (default: detections.csv)
--test-camera              Test camera/source and exit
--no-object-detection      Disable YOLOv8 (faster processing)
--display                  Show live display window (Windows/Linux)
--headless                 Disable display
```

## ⚡ Performance Tips

```bash
# Disable object detection for speed
python face_detection_windows.py --camera 0 --no-object-detection

# Use headless mode for faster processing
python face_detection_windows.py --video stream.mp4 --headless

# Test camera first
python face_detection_windows.py --test-camera --camera 0
```

## 🐛 Troubleshooting

### Camera Not Found
```bash
python face_detection_windows.py --test-camera
```

### No Display on Windows
- Install Pillow: `pip install Pillow`
- Use `--display` flag

### No Display on Linux/WSL
- Use `--headless` flag
- Or configure X11 forwarding

### IP Camera Connection Failed
- Check IP address is correct
- Ensure device on same network
- Verify camera app is running
- Use `--test-camera` to validate URL

## 📁 Project Structure

```
Face_Detection/
├── face_detection.py              # Cross-platform version
├── face_detection_windows.py      # Windows-optimized
├── face_detection_linux.py        # Linux-optimized
├── detection_logger.py            # CSV logging module
├── requirements.txt               # Python dependencies
├── PLATFORM_GUIDE.md             # Detailed guides
├── README.md                      # This file
├── known_faces/                   # Add face images here
├── detections.csv                 # Detection log
└── tests/                         # Unit tests
```

## 💾 System Requirements

### Minimum
- CPU: Dual-core 2GHz
- RAM: 4GB
- Storage: 500MB

### Recommended
- CPU: Quad-core 3GHz+
- RAM: 8GB+
- GPU: NVIDIA CUDA (optional, for YOLOv8)

## 📦 Dependencies

```
opencv-python>=4.8.0       # Computer vision
face-recognition>=1.3.0    # Face recognition
numpy>=1.24.0             # Numerical computing
dlib>=19.24.0             # Machine learning
Pillow>=10.0.0            # Image processing
ultralytics>=8.0.0        # YOLOv8 object detection
```

## 🎯 Use Cases

- 🏢 Access control & security
- 📊 Crowd counting & analytics
- 🎓 Attendance tracking
- 🚔 Surveillance systems
- 📱 Mobile device detection
- 🎬 Video analysis & processing

## 🔐 Privacy

- All processing is local (no cloud upload)
- Known faces stored locally only
- CSV log contains only timestamps and names
- No video files automatically saved

## 📞 Support & Contributing

- Check **PLATFORM_GUIDE.md** for detailed help
- See **Troubleshooting** section above
- Open an issue on GitHub

## 📝 License

See LICENSE file for details

---

**Made with ❤️ for real-time detection**

Last Updated: November 28, 2025

### Command Line Options

```
--known-faces, -k   Directory containing known face images (default: known_faces)
--log-file, -l      Path to the detection log CSV file (default: detections.csv)
--camera, -c        Camera index to use (default: 0)
```

### Example

```bash
python face_detection.py --known-faces ./my_faces --log-file ./logs/detections.csv --camera 1
```

## Output

### Video Display

- Green rectangle: Person wearing a mask
- Red rectangle: Person not wearing a mask
- Label shows the person's name (or "Unknown") and mask status

### CSV Log

The application creates a CSV file with the following columns:
- `timestamp`: Date and time of detection
- `name`: Name of the detected person (or "Unknown")
- `mask_status`: "Mask", "No Mask", or "Unknown"
- `confidence`: Confidence score for face recognition

## Project Structure

```
Face_Detection/
├── face_detection.py     # Main application
├── detection_logger.py   # CSV logging module
├── known_faces/          # Directory for known face images
├── requirements.txt      # Python dependencies
├── LICENSE               # GPL-3.0 license
└── README.md             # This file
```

## Controls

- Press `q` to quit the application

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.