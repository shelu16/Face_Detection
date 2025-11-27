# Face Detection

A real-time face recognition application that detects faces, recognizes known individuals, checks for mask-wearing, and logs all detections to a CSV file.

## Features

- **Face Detection**: Detects faces in real-time using the webcam
- **Face Recognition**: Recognizes known individuals from pre-loaded images
- **Mask Detection**: Identifies whether a person is wearing a face mask
- **CSV Logging**: Logs all detections with timestamp, name, mask status, and confidence score

## Requirements

- Python 3.8+
- Webcam or camera device

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shelu16/Face_Detection.git
cd Face_Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: On some systems, you may need to install additional system packages for dlib:
```bash
# Ubuntu/Debian
sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev libx11-dev

# macOS
brew install cmake
```

## Usage

### Adding Known Faces

Add images of people you want to recognize to the `known_faces` directory. Name each image file with the person's name (e.g., `john_doe.jpg`).

### Running the Application

```bash
python face_detection.py
```

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