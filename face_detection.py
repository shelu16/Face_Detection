"""
Real-time Face Recognition Application
Detects faces, recognizes known individuals, checks for mask-wearing,
and logs all detections to a CSV file.
"""

import argparse
import os
import time

import cv2
import face_recognition
import numpy as np

from detection_logger import DetectionLogger

# Detection thresholds
FACE_RECOGNITION_THRESHOLD = 0.6  # Distance threshold for face matching
MASK_DETECTION_THRESHOLD = 0.25   # Color coverage threshold for mask detection


class FaceRecognitionApp:
    """Real-time face recognition application with mask detection."""

    def __init__(self, known_faces_dir="known_faces", log_file="detections.csv"):
        """
        Initialize the face recognition application.

        Args:
            known_faces_dir: Directory containing known face images.
            log_file: Path to the CSV log file.
        """
        self.known_faces_dir = known_faces_dir
        self.known_face_encodings = []
        self.known_face_names = []
        self.logger = DetectionLogger(log_file)

        # Load Haar cascade for face detection (used for mask detection)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

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
        Process a single frame for face detection and recognition.

        Args:
            frame: The video frame to process.

        Returns:
            Processed frame with annotations.
        """
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

    def run(self, camera_index=0):
        """
        Run the real-time face recognition application.

        Args:
            camera_index: Index of the camera to use.
        """
        print("Starting face recognition application...")
        print(f"Logging detections to: {self.logger.get_log_path()}")
        print("Press 'q' to quit")

        video_capture = cv2.VideoCapture(camera_index)

        if not video_capture.isOpened():
            print("Error: Could not open video capture device")
            return

        try:
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    print("Error: Could not read frame")
                    break

                # Process the frame
                processed_frame = self.process_frame(frame)

                # Display the result
                cv2.imshow('Face Recognition', processed_frame)

                # Check for quit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            video_capture.release()
            cv2.destroyAllWindows()
            print(f"\nDetection log saved to: {self.logger.get_log_path()}")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='Real-time Face Recognition with Mask Detection')
    parser.add_argument('--known-faces', '-k', default='known_faces',
                        help='Directory containing known face images')
    parser.add_argument('--log-file', '-l', default='detections.csv',
                        help='Path to the detection log CSV file')
    parser.add_argument('--camera', '-c', type=int, default=0,
                        help='Camera index to use')

    args = parser.parse_args()

    app = FaceRecognitionApp(
        known_faces_dir=args.known_faces,
        log_file=args.log_file
    )
    app.run(camera_index=args.camera)


if __name__ == "__main__":
    main()
