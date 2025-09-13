#!/usr/bin/env python3
"""
Quick FPS test without GUI to isolate performance issues
"""

import cv2
import time
import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_headless_fps():
    """Test FPS without GUI components"""
    print("🔬 Testing headless FPS performance...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    # DON'T set camera properties - use native settings
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Set only buffer size which is safe
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    frame_count = 0
    start_time = time.time()
    test_duration = 10  # 10 seconds
    
    print(f"⏱️ Running headless test for {test_duration} seconds...")
    print("   Press 'q' in the video window to stop early")
    
    while time.time() - start_time < test_duration:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Don't resize - use native camera resolution for best performance
        # The camera diagnostic showed it works at 29 FPS at native 1280x720
        
        # Add FPS counter (minimal processing)
        current_time = time.time()
        elapsed = current_time - start_time
        if elapsed > 0:
            current_fps = frame_count / elapsed
            cv2.putText(frame, f"FPS: {current_fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame (minimal GUI)
        cv2.imshow("Headless FPS Test", frame)
        
        # Non-blocking key check
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    elapsed = time.time() - start_time
    final_fps = frame_count / elapsed
    
    print(f"\\n📊 Headless Test Results:")
    print(f"   Total frames: {frame_count}")
    print(f"   Duration: {elapsed:.1f}s")
    print(f"   Average FPS: {final_fps:.1f}")
    
    if final_fps >= 20:
        print("✅ EXCELLENT: Camera and basic processing >20 FPS")
    elif final_fps >= 15:
        print("✅ GOOD: Camera performance 15-20 FPS")
    else:
        print("⚠️ ISSUE: Camera performance <15 FPS - hardware limitation?")
    
    return final_fps

if __name__ == "__main__":
    try:
        test_headless_fps()
    except KeyboardInterrupt:
        print("\\n⚠️ Test interrupted")
    except Exception as e:
        print(f"\\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()