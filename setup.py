#!/usr/bin/env python3
"""
Setup script for Face Detection application
Helps users configure and test their setup
"""

import os
import sys
import platform
import subprocess


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def check_python():
    """Check Python version"""
    print_header("🐍 Python Version Check")
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Python version: {version}")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher required!")
        return False
    print("✓ Python version OK\n")
    return True


def check_platform():
    """Detect operating system"""
    print_header("💻 Platform Detection")
    os_name = platform.system()
    print(f"Detected OS: {os_name}")
    
    if os_name == "Windows":
        print("→ Use: python face_detection_windows.py")
    elif os_name == "Linux":
        if "microsoft" in platform.release().lower():
            print("→ Detected: Windows Subsystem for Linux (WSL)")
            print("→ Use: python face_detection_linux.py --headless")
        else:
            print("→ Use: python face_detection_linux.py")
    elif os_name == "Darwin":
        print("→ macOS detected (use Linux version)")
        print("→ Use: python face_detection.py")
    
    print()
    return os_name


def check_dependencies():
    """Check installed Python packages"""
    print_header("📦 Dependency Check")
    
    required = {
        'cv2': 'opencv-python',
        'face_recognition': 'face-recognition',
        'numpy': 'numpy',
    }
    
    optional = {
        'ultralytics': 'ultralytics (YOLOv8)',
        'PIL': 'Pillow',
    }
    
    missing_required = []
    missing_optional = []
    
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_required.append(package)
    
    print()
    for module, package in optional.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"⚠ {package} (optional)")
            missing_optional.append(package)
    
    print()
    
    if missing_required:
        print("Missing required packages. Install with:")
        print(f"  pip install {' '.join(missing_required)}")
        print()
        return False
    
    if missing_optional:
        print("Optional packages missing. Install with:")
        print(f"  pip install {' '.join(missing_optional)}")
        print()
    
    return True


def check_directories():
    """Check/create required directories"""
    print_header("📁 Directory Check")
    
    dirs = {
        'known_faces': 'Known face images',
    }
    
    for dir_name, description in dirs.items():
        if os.path.exists(dir_name):
            count = len([f for f in os.listdir(dir_name) if f.endswith(('.jpg', '.png', '.jpeg'))])
            print(f"✓ {dir_name}/ ({description}) - {count} images found")
        else:
            os.makedirs(dir_name, exist_ok=True)
            print(f"✓ Created {dir_name}/ ({description})")
    
    print()


def test_camera():
    """Test camera connection"""
    print_header("📷 Camera Test")
    
    try:
        import cv2
        
        print("Testing camera indices 0-3...")
        cameras = []
        
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    cameras.append(i)
                    print(f"✓ Camera {i} - Available")
                else:
                    print(f"⚠ Camera {i} - Found but can't read")
            else:
                print(f"✗ Camera {i} - Not available")
        
        print()
        if cameras:
            print(f"Found {len(cameras)} working camera(s): {cameras}")
            print(f"Use: --camera {cameras[0]}")
        else:
            print("No cameras found. Use IP camera or video file instead.")
        print()
        return len(cameras) > 0
        
    except ImportError:
        print("❌ OpenCV not installed. Install with: pip install opencv-python")
        print()
        return False


def main():
    """Main setup function"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Face Detection & Recognition Setup".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Run checks
    checks = [
        ("Python Version", check_python),
        ("Platform Detection", check_platform),
        ("Dependencies", check_dependencies),
        ("Directories", check_directories),
        ("Camera", test_camera),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"Error during {name} check: {e}\n")
            results[name] = False
    
    # Summary
    print_header("📋 Setup Summary")
    
    if results.get("Python Version") and results.get("Dependencies"):
        print("✅ Setup looks good! You can now run:")
        
        os_name = platform.system()
        if os_name == "Windows":
            print("\n  python face_detection_windows.py --camera 0 --display")
        elif os_name == "Linux":
            if "microsoft" in platform.release().lower():
                print("\n  python face_detection_linux.py --camera 0 --headless")
            else:
                print("\n  python face_detection_linux.py --camera 0")
        
        print("\nOr use IP camera:")
        print("  python face_detection_windows.py --video http://192.168.1.100:4747/video --display")
        print("\nAdd face images to 'known_faces/' directory for recognition.\n")
    else:
        print("❌ Please fix the issues above before running the application.\n")
        if not results.get("Dependencies"):
            print("Install dependencies with:")
            print("  pip install -r requirements.txt")
            if results.get("Platform Detection"):
                print("  pip install ultralytics  # For object detection\n")


if __name__ == "__main__":
    main()
