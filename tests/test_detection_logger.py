"""Tests for the detection logger module."""

import csv
import os
import tempfile
import unittest
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection_logger import DetectionLogger


class TestDetectionLogger(unittest.TestCase):
    """Test cases for DetectionLogger class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, 'test_detections.csv')

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        os.rmdir(self.temp_dir)

    def test_init_creates_csv_with_headers(self):
        """Test that initialization creates a CSV file with proper headers."""
        logger = DetectionLogger(self.log_file)

        self.assertTrue(os.path.exists(self.log_file))

        with open(self.log_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            self.assertEqual(headers, ['timestamp', 'name', 'mask_status', 'confidence'])

    def test_log_detection_basic(self):
        """Test basic detection logging."""
        logger = DetectionLogger(self.log_file)
        logger.log_detection('John Doe', 'Mask', 0.95)

        with open(self.log_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            row = next(reader)

        self.assertEqual(row[1], 'John Doe')
        self.assertEqual(row[2], 'Mask')
        self.assertEqual(row[3], '0.95')

    def test_log_detection_without_confidence(self):
        """Test detection logging without confidence score."""
        logger = DetectionLogger(self.log_file)
        logger.log_detection('Unknown', 'No Mask')

        with open(self.log_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            row = next(reader)

        self.assertEqual(row[1], 'Unknown')
        self.assertEqual(row[2], 'No Mask')
        self.assertEqual(row[3], '')

    def test_log_detection_multiple_entries(self):
        """Test multiple detection logs."""
        logger = DetectionLogger(self.log_file)
        logger.log_detection('Person A', 'Mask', 0.9)
        logger.log_detection('Person B', 'No Mask', 0.85)
        logger.log_detection('Unknown', 'Unknown')

        with open(self.log_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 4)  # Header + 3 entries

    def test_log_detection_timestamp_format(self):
        """Test that timestamps are in correct format."""
        logger = DetectionLogger(self.log_file)
        logger.log_detection('Test', 'Mask')

        with open(self.log_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            row = next(reader)

        timestamp = row[0]
        # Should be in format 'YYYY-MM-DD HH:MM:SS'
        try:
            datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            self.fail(f"Timestamp '{timestamp}' is not in expected format")

    def test_get_log_path(self):
        """Test that get_log_path returns absolute path."""
        logger = DetectionLogger(self.log_file)
        path = logger.get_log_path()

        self.assertTrue(os.path.isabs(path))
        self.assertTrue(path.endswith('test_detections.csv'))

    def test_append_to_existing_file(self):
        """Test that new logger appends to existing file."""
        logger1 = DetectionLogger(self.log_file)
        logger1.log_detection('Person 1', 'Mask')

        logger2 = DetectionLogger(self.log_file)
        logger2.log_detection('Person 2', 'No Mask')

        with open(self.log_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 3)  # Header + 2 entries


if __name__ == '__main__':
    unittest.main()
