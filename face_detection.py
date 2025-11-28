"""
Real-time Face Recognition Application
Detects faces, recognizes known individuals, checks for mask-wearing,
logs all detections, and performs object detection using YOLOv8.
"""

import argparse
import os
import platform
import re
import threading
import time
from urllib.parse import urlparse
from PIL import Image, ImageTk
import tkinter as tk

import cv2
import face_recognition
import numpy as np

from detection_logger import DetectionLogger

# Try to import YOLOv8 for object detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLOv8 not installed. Object detection disabled.")
    print("Install with: pip install ultralytics")

# Detection thresholds
FACE_RECOGNITION_THRESHOLD = 0.6  # Distance threshold for face matching
MASK_DETECTION_THRESHOLD = 0.25   # Color coverage threshold for mask detection
YOLO_CONFIDENCE_THRESHOLD = 0.5   # YOLOv8 confidence threshold

# Platform detection
PLATFORM = platform.system()  # 'Linux', 'Windows', 'Darwin'
IS_WSL = 'microsoft' in platform.release().lower()  # Check if running on WSL


def is_valid_ip_camera_url(url):
    """
    Validate if the URL is a valid IP camera stream.
    
    Args:
        url: URL string to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not url:
        return False, "URL cannot be empty"
    
    # Check for common IP camera URL patterns
    url_patterns = [
        r'^https?://',  # HTTP/HTTPS
        r'^rtsp://',    # RTSP
        r'^rtmp://',    # RTMP
    ]
    
    is_url = any(re.match(pattern, url) for pattern in url_patterns)
    
    if not is_url:
        return False, "Invalid URL format. Use http://, https://, rtsp://, or rtmp://"
    
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return False, "URL must contain a valid network location (host:port)"
        return True, None
    except Exception as e:
        return False, f"Error parsing URL: {str(e)}"


def test_camera_source(source):
    """
    Test if a camera source (index, file path, or URL) is accessible.
    
    Args:
        source: Camera index (int), file path (str), or URL (str)
        
    Returns:
        Tuple of (is_accessible, error_message)
    """
    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return False, "Could not open camera source"
        
        # Try to read one frame
        ret, _ = cap.read()
        cap.release()
        
        if not ret:
            return False, "Could not read frame from camera source"
        
        return True, None
    except Exception as e:
        return False, f"Error testing camera: {str(e)}"


class DisplayHelper:
    """Helper class to manage display window with cross-platform support using Tkinter/PIL."""
    
    def __init__(self, window_name="Face Recognition", width=800, height=600):
        """
        Initialize display helper.
        
        Args:
            window_name: Name of the display window
            width: Window width
            height: Window height
        """
        self.window_name = window_name
        self.width = width
        self.height = height
        self.display_available = False
        self.root = None
        self.label = None
        self.quit_requested = False
        self._init_window()
    
    def _init_window(self):
        """Initialize Tkinter window."""
        try:
            self.root = tk.Tk()
            self.root.title(self.window_name)
            self.root.geometry(f"{self.width}x{self.height}")
            
            # Create label for displaying frames
            self.label = tk.Label(self.root, bg="black")
            self.label.pack(fill=tk.BOTH, expand=True)
            
            # Bind quit event
            self.root.protocol("WM_DELETE_WINDOW", self._on_quit)
            self.root.bind('q', lambda e: self._on_quit())
            
            self.display_available = True
        except Exception as e:
            print(f"Warning: Could not initialize display - {e}")
            self.display_available = False
            self._cleanup_window()
    
    def _on_quit(self):
        """Handle quit request."""
        self.quit_requested = True
    
    def _cleanup_window(self):
        """Clean up Tkinter window."""
        try:
            if self.root:
                self.root.destroy()
                self.root = None
        except Exception:
            pass
    
    def show(self, frame):
        """
        Display frame in a window.
        
        Args:
            frame: Image frame to display (BGR format from OpenCV)
            
        Returns:
            True if display succeeded, False otherwise
        """
        if not self.display_available or not self.root:
            return False
        
        try:
            # Update window events (non-blocking)
            self.root.update()
            
            # Convert BGR to RGB and resize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (self.width, self.height))
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Convert to PhotoImage for Tkinter
            tk_image = ImageTk.PhotoImage(pil_image)
            
            # Update label
            self.label.config(image=tk_image)
            self.label.image = tk_image  # Keep a reference to prevent garbage collection
            
            return True
        except Exception as e:
            print(f"Error displaying frame: {e}")
            self.display_available = False
            return False
    
    def check_quit(self, delay=1):
        """
        Check for quit command.
        
        Args:
            delay: Delay in milliseconds (ignored for Tkinter)
            
        Returns:
            True if quit command pressed, False otherwise
        """
        if not self.display_available:
            return False
        
        return self.quit_requested
    
    def close(self):
        """Close display window."""
        self._cleanup_window()
        self.display_available = False


class FaceRecognitionApp:
    """Real-time face recognition application with mask detection and object detection."""

    def __init__(self, known_faces_dir="known_faces", log_file="detections.csv", enable_object_detection=True):
        """
        Initialize the face recognition application.

        Args:
            known_faces_dir: Directory containing known face images.
            log_file: Path to the CSV log file.
            enable_object_detection: Enable YOLOv8 object detection.
        """
        self.known_faces_dir = known_faces_dir
        self.known_face_encodings = []
        self.known_face_names = []
        self.logger = DetectionLogger(log_file)

        # Load Haar cascade for face detection (used for mask detection)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Load YOLOv8 model for object detection
        self.yolo_model = None
        self.object_detection_enabled = False
        if enable_object_detection and YOLO_AVAILABLE:
            try:
                print("Loading YOLOv8 model for object detection...")
                self.yolo_model = YOLO('yolov8n.pt')  # nano model for speed
                self.object_detection_enabled = True
                print("YOLOv8 model loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load YOLOv8 model - {e}")
                self.object_detection_enabled = False

        # Load known faces
        self._load_known_faces()

        # Track recently logged faces to avoid duplicate logs
        self.recently_logged = {}
        self.log_cooldown = 5  # seconds between logs for same person

    def _load_known_faces(self):
        """Load and encode known faces from the known_faces directory."""
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
            print(f"Created directory: {self.known_faces_dir}")
            print("Add images of known faces to this directory (named as person_name.jpg)")
            return

        for filename in os.listdir(self.known_faces_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(self.known_faces_dir, filename)
                try:
                    image = face_recognition.load_image_file(filepath)
                    encodings = face_recognition.face_encodings(image)

                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        # Use filename without extension as the name
                        name = os.path.splitext(filename)[0]
                        self.known_face_names.append(name)
                        print(f"Loaded face: {name}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

        print(f"Loaded {len(self.known_face_names)} known faces")

    def _detect_mask(self, frame, face_location):
        """
        Detect if a person is wearing a mask.

        This uses a simple heuristic based on the lower face region.
        For production use, consider using a trained mask detection model.

        Args:
            frame: The video frame.
            face_location: Tuple of (top, right, bottom, left) coordinates.

        Returns:
            String indicating mask status: 'Mask', 'No Mask', or 'Unknown'
        """
        top, right, bottom, left = face_location
        face_height = bottom - top

        # Get the lower half of the face (nose and mouth region)
        lower_face_top = top + int(face_height * 0.5)
        lower_face = frame[lower_face_top:bottom, left:right]

        if lower_face.size == 0:
            return "Unknown"

        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(lower_face, cv2.COLOR_BGR2HSV)

        # Check for common mask colors (blue, white, black, green)
        # Blue surgical masks
        blue_lower = np.array([90, 50, 50])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

        # White masks
        white_lower = np.array([0, 0, 180])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)

        # Black masks
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 255, 50])
        black_mask = cv2.inRange(hsv, black_lower, black_upper)

        # Calculate mask coverage percentages
        total_pixels = lower_face.shape[0] * lower_face.shape[1]
        if total_pixels == 0:
            return "Unknown"

        blue_ratio = cv2.countNonZero(blue_mask) / total_pixels
        white_ratio = cv2.countNonZero(white_mask) / total_pixels
        black_ratio = cv2.countNonZero(black_mask) / total_pixels

        # If significant mask color is detected in lower face
        if (blue_ratio > MASK_DETECTION_THRESHOLD or
                white_ratio > MASK_DETECTION_THRESHOLD or
                black_ratio > MASK_DETECTION_THRESHOLD):
            return "Mask"

        return "No Mask"

    def _detect_objects(self, frame):
        """
        Detect objects in the frame using YOLOv8.

        Args:
            frame: The video frame.

        Returns:
            List of detected objects with their info.
        """
        if not self.object_detection_enabled or not self.yolo_model:
            return []

        try:
            # Run YOLOv8 inference
            results = self.yolo_model(frame, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)
            
            detections = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = self.yolo_model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'box': (x1, y1, x2, y2)
                    })
            
            return detections
        except Exception as e:
            # Silently fail if object detection fails
            return []

    def _should_log(self, name):
        """
        Check if we should log this detection (avoid duplicate logs).

        Args:
            name: Name of the detected person.

        Returns:
            Boolean indicating whether to log.
        """
        current_time = time.time()

        if name in self.recently_logged:
            if current_time - self.recently_logged[name] < self.log_cooldown:
                return False

        self.recently_logged[name] = current_time
        return True

    def process_frame(self, frame):
        """
        Process a single frame for face detection, recognition, mask detection, and object detection.

        Args:
            frame: The video frame to process.

        Returns:
            Processed frame with annotations.
        """
        # Detect objects using YOLOv8
        objects = self._detect_objects(frame)
        
        # Draw object detections
        for obj in objects:
            x1, y1, x2, y2 = obj['box']
            confidence = obj['confidence']
            class_name = obj['class']
            
            # Draw bounding box (cyan for objects)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            
            # Draw label
            label = f"{class_name} ({confidence:.2f})"
            cv2.rectangle(frame, (x1, y1 - 30), (x1 + len(label) * 8, y1), (255, 255, 0), cv2.FILLED)
            cv2.putText(frame, label, (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find faces in the frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale back up face locations
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Recognize face
            name = "Unknown"
            confidence = 0.0

            if self.known_face_encodings:
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
                )
                best_match_index = np.argmin(face_distances)
                confidence = 1 - face_distances[best_match_index]

                if face_distances[best_match_index] < FACE_RECOGNITION_THRESHOLD:
                    name = self.known_face_names[best_match_index]

            # Detect mask
            mask_status = self._detect_mask(frame, (top, right, bottom, left))

            # Log detection
            if self._should_log(name):
                self.logger.log_detection(name, mask_status, round(confidence, 2))

            # Draw rectangle around face
            color = (0, 255, 0) if mask_status == "Mask" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw label
            label = f"{name} - {mask_status}"
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        return frame

    def run(self, camera_index=1, video_file=None, headless=None, display=False):
        """
        Run the real-time face recognition application.

        Args:
            camera_index: Index of the camera to use (if video_file is None).
            video_file: Path to a video file to process instead of camera.
            headless: If True, process without displaying video. 
                     If None (default), auto-detect based on platform/environment.
            display: If True, try to show display window (overrides headless auto-detection).
        """
        # Auto-detect headless mode if not explicitly set
        if display:
            headless = False
        elif headless is None:
            # Use headless mode on WSL by default (unless running with X11 forwarding)
            if IS_WSL:
                headless = not os.environ.get('DISPLAY')
            else:
                # On native Windows, default to headless (no GTK+ support by default)
                headless = PLATFORM == 'Windows'
        
        # Initialize display helper
        display_helper = DisplayHelper("Face Recognition", width=800, height=600)
        
        print("Starting face recognition application...")
        print(f"Platform: {PLATFORM}")
        if IS_WSL:
            print("Running on WSL (Windows Subsystem for Linux)")
        print(f"Logging detections to: {self.logger.get_log_path()}")
        
        # Determine actual display mode
        use_display = not headless and display_helper.display_available
        
        if use_display:
            print("Display window enabled")
            print("Press 'q' to quit")
        else:
            print("Running in headless mode (no display)")

        if video_file:
            video_capture = cv2.VideoCapture(video_file)
            source_type = f"video file: {video_file}"
        else:
            video_capture = cv2.VideoCapture(camera_index)
            source_type = f"camera (index: {camera_index})"

        if not video_capture.isOpened():
            print(f"Error: Could not open {source_type}")
            return

        try:
            frame_count = 0
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    print(f"Finished processing. Total frames: {frame_count}")
                    break

                frame_count += 1
                # Process the frame
                processed_frame = self.process_frame(frame)

                # Display the result if display is enabled
                if use_display:
                    if display_helper.show(processed_frame):
                        if display_helper.check_quit(delay=1):
                            print("Quit command received")
                            break
                    else:
                        # Display failed, switch to headless
                        print("\nWarning: Display failed, continuing in headless mode...")
                        use_display = False
                else:
                    # Print progress in headless mode
                    if frame_count % 30 == 0:
                        print(f"Processed {frame_count} frames...")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Cleaning up...")
        finally:
            try:
                video_capture.release()
            except Exception:
                pass  # Ignore errors during release
            
            display_helper.close()
            
            print(f"Detection log saved to: {self.logger.get_log_path()}")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description='Real-time Face Recognition with Mask Detection',
        epilog='''
EXAMPLES:
  # Use local webcam (index 0)
  python face_detection.py --camera 0
  
  # Use video file
  python face_detection.py --video path/to/video.mp4
  
  # Use IP camera (e.g., IP Webcam app on phone)
  python face_detection.py --video http://192.168.1.100:8080/video
  
  # Use RTSP stream from mobile device or IP camera
  python face_detection.py --video rtsp://192.168.1.100:554/stream
  
  # Force display mode (try to show video window)
  python face_detection.py --camera 0 --display
  
  # Force headless mode (no display, faster processing)
  python face_detection.py --video http://192.168.1.100:8080/video --headless
  
MOBILE CAMERA SETUP:
  1. Install "IP Webcam" app on your Android phone (Google Play Store)
  2. Open the app and note the IP address (e.g., 192.168.1.100)
  3. Run: python face_detection.py --video http://192.168.1.100:8080/video
  
  For iPhone: Use apps like "Codeshot" or "Reincubate Codeshot"
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--known-faces', '-k', default='known_faces',
                        help='Directory containing known face images')
    parser.add_argument('--log-file', '-l', default='detections.csv',
                        help='Path to the detection log CSV file')
    parser.add_argument('--camera', '-c', type=int, default=1,
                        help='Camera index to use (0, 1, 2, etc.)')
    parser.add_argument('--video', '-v', default=None,
                        help='Video source: file path, IP camera URL (http://...), or RTSP stream (rtsp://...)')
    parser.add_argument('--test-camera', action='store_true',
                        help='Test camera/video source and exit')
    parser.add_argument('--headless', action='store_true',
                        help='Force headless mode (no display). Auto-detected on WSL if not set.')
    parser.add_argument('--display', action='store_true',
                        help='Force display mode (show video). Overrides auto-detection.')
    parser.add_argument('--no-object-detection', action='store_true',
                        help='Disable YOLOv8 object detection (faster processing)')

    args = parser.parse_args()

    # Test camera source if requested
    if args.test_camera:
        if args.video:
            print(f"Testing video source: {args.video}")
            # Validate URL if it looks like an IP camera
            if args.video.startswith(('http://', 'https://', 'rtsp://', 'rtmp://')):
                is_valid, error = is_valid_ip_camera_url(args.video)
                if not is_valid:
                    print(f"❌ Invalid URL: {error}")
                    return
                print("✓ URL format validated")
            
            is_accessible, error = test_camera_source(args.video)
            if is_accessible:
                print(f"✓ Successfully connected to: {args.video}")
            else:
                print(f"❌ Error: {error}")
        else:
            print(f"Testing camera index: {args.camera}")
            is_accessible, error = test_camera_source(args.camera)
            if is_accessible:
                print(f"✓ Camera {args.camera} is accessible")
            else:
                print(f"❌ Error: {error}")
        return

    # Determine headless mode
    headless_mode = None
    if args.display:
        headless_mode = False
    elif args.headless:
        headless_mode = True
    # else: None (auto-detect)

    # Validate video URL if provided
    if args.video and args.video.startswith(('http://', 'https://', 'rtsp://', 'rtmp://')):
        is_valid, error = is_valid_ip_camera_url(args.video)
        if not is_valid:
            print(f"Error: Invalid video URL - {error}")
            print("Valid formats: http://..., https://..., rtsp://..., rtmp://...")
            return

    app = FaceRecognitionApp(
        known_faces_dir=args.known_faces,
        log_file=args.log_file,
        enable_object_detection=not args.no_object_detection
    )
    app.run(camera_index=args.camera, video_file=args.video, headless=headless_mode, display=args.display)


if __name__ == "__main__":
    main()
