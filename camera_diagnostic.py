#!/usr/bin/env python3
"""
Camera diagnostic tool
"""

import cv2
import time

def diagnose_camera():
    """Diagnose camera issues"""
    print("🔍 Camera Diagnostic Tool")
    print("=" * 30)
    
    # Test different camera indices
    for i in range(3):
        print(f"\\nTesting camera index {i}...")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            print(f"✅ Camera {i} opened successfully")
            
            # Get camera properties
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"   Resolution: {width}x{height}")
            print(f"   FPS: {fps}")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                print(f"   Frame size: {frame.shape}")
                print("   ✅ Frame read successful")
                
                # Quick FPS test
                frame_count = 0
                start_time = time.time()
                
                for _ in range(30):  # Read 30 frames
                    ret, frame = cap.read()
                    if ret:
                        frame_count += 1
                    else:
                        break
                
                elapsed = time.time() - start_time
                if elapsed > 0 and frame_count > 0:
                    fps_test = frame_count / elapsed
                    print(f"   Actual FPS: {fps_test:.1f}")
                else:
                    print("   ❌ Could not measure FPS")
            else:
                print("   ❌ Could not read frame")
            
            cap.release()
        else:
            print(f"❌ Camera {i} could not be opened")
    
    print("\\n🔧 Recommended camera settings test...")
    
    # Test with specific settings
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        # Set optimal settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("Settings applied:")
        print(f"   Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        print(f"   Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"   FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        print(f"   Buffer: {cap.get(cv2.CAP_PROP_BUFFERSIZE)}")
        
        # Performance test with settings
        frame_count = 0
        start_time = time.time()
        test_duration = 3
        
        print(f"\\n⏱️ Performance test ({test_duration}s)...")
        
        while time.time() - start_time < test_duration:
            ret, frame = cap.read()
            if ret:
                frame_count += 1
            else:
                print(f"   Frame read failed at frame {frame_count}")
                break
        
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        print(f"✅ Performance test complete:")
        print(f"   Frames: {frame_count}")
        print(f"   Duration: {elapsed:.1f}s")
        print(f"   FPS: {fps:.1f}")
        
        if fps >= 15:
            print("✅ Camera performance is good")
        else:
            print("⚠️ Camera performance is low")
        
        cap.release()

if __name__ == "__main__":
    try:
        diagnose_camera()
    except Exception as e:
        print(f"❌ Diagnostic error: {e}")
        import traceback
        traceback.print_exc()