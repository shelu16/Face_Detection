"""
Detection Logger Module
Handles CSV logging for face detection events.
"""

import csv
import os
from datetime import datetime


class DetectionLogger:
    """Logger for face detection events with CSV output."""

    def __init__(self, log_file="detections.csv"):
        """
        Initialize the detection logger.

        Args:
            log_file: Path to the CSV log file.
        """
        self.log_file = log_file
        self._initialize_csv()

    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'name', 'mask_status', 'confidence'])

    def log_detection(self, name, mask_status, confidence=None):
        """
        Log a face detection event.

        Args:
            name: Name of the detected person or 'Unknown'.
            mask_status: Whether the person is wearing a mask ('Mask', 'No Mask', 'Unknown').
            confidence: Optional confidence score for the detection.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, name, mask_status, confidence if confidence else ''])

    def get_log_path(self):
        """Return the path to the log file."""
        return os.path.abspath(self.log_file)
