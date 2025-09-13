#!/usr/bin/env python3
"""
Debug camera reading issue in main app
"""

import cv2
import time
import sys
import os

def test_camera_reading():
    """Test camera reading similar to main app"""
    print("🔍 Debugging camera reading issue...")
    print("=" * 50)
    
    try:
        # Initialize camera exactly like main app
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Failed to open camera")
            return False
        
        # Set same properties as main app
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        
        print("✅ Camera initialized with same settings as main app")
        
        # Test reading frames
        frame_count = 0
        max_frames = 10
        
        print(f"📹 Testing frame reading ({max_frames} frames)...")
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                print(f"❌ Failed to read frame {frame_count + 1}")
                print(f"   ret = {ret}")
                break
            
            print(f"✅ Frame {frame_count + 1}: {frame.shape}")
            frame_count += 1
            
            # Small delay like waitKey
            time.sleep(0.033)  # ~30fps
        
        cap.release()
        
        if frame_count == max_frames:
            print(f"🎉 Successfully read all {max_frames} frames!")
            return True
        else:
            print(f"⚠️  Only read {frame_count}/{max_frames} frames")
            return False
            
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        return False

def test_with_display():
    """Test with OpenCV display like main app"""
    print("\n🖥️  Testing with OpenCV display window...")
    print("=" * 50)
    
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Failed to open camera")
            return False
        
        # Same settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        
        print("✅ Camera initialized")
        print("📹 Showing camera feed - press 'q' to quit")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print(f"❌ Failed to read frame after {frame_count} successful frames")
                break
            
            frame_count += 1
            
            # Add frame counter to image
            cv2.putText(frame, f"Frame: {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Camera Test", frame)
            
            # Check for quit - exactly like main app
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print(f"✅ User quit after {frame_count} frames")
                break
                
            # Auto-quit after 100 frames for testing
            if frame_count >= 100:
                print(f"✅ Auto-quit after {frame_count} frames")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return frame_count > 0
        
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Camera Reading Debug Test")
    print("=" * 50)
    
    # Test 1: Basic reading
    success1 = test_camera_reading()
    
    # Test 2: With display
    success2 = test_with_display()
    
    print("\n" + "=" * 50)
    print("📊 RESULTS:")
    print(f"  Basic reading test: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"  Display test:       {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("\n🎉 Camera is working perfectly!")
        print("💡 The issue might be in the main app logic")
    else:
        print("\n⚠️  Camera has issues - need to investigate further")