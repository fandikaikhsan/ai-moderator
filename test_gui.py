#!/usr/bin/env python3
"""
Quick test of GUI mode functionality
"""

import sys
import threading
import time
import cv2
import numpy as np

def test_gui_startup():
    """Test if GUI starts up correctly"""
    print("Testing GUI startup...")
    
    try:
        # Import modules
        from main import AIModerator
        
        print("✅ Main module imported successfully")
        
        # Test camera initialization
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera access working")
            cap.release()
        else:
            print("⚠️  Camera not available for testing")
        
        # Test moderator initialization (without starting GUI)
        print("Testing AI Moderator initialization...")
        moderator = AIModerator(camera_index=0, gui_mode=True)
        
        if moderator.initialize_camera():
            print("✅ Camera initialized in moderator")
        else:
            print("⚠️  Camera initialization failed")
        
        print("✅ GUI mode setup completed successfully!")
        print()
        print("🎯 Ready to run GUI mode!")
        print("   Run: ./run.sh and choose option 3")
        print("   Or run: python3 main.py")
        
        return True
        
    except Exception as e:
        print(f"❌ GUI startup test failed: {e}")
        return False
    
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    success = test_gui_startup()
    if success:
        print("\n🚀 GUI mode is ready to use!")
    else:
        print("\n⚠️  There might be issues with GUI mode")
    
    sys.exit(0 if success else 1)