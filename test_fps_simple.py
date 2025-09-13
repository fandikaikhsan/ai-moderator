#!/usr/bin/env python3
"""
Simple FPS test for AI Moderator optimization verification
"""

import time
import cv2
import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_camera_only():
    """Test baseline camera performance"""
    print("🎥 Testing baseline camera performance...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return 0
    
    frame_count = 0
    start_time = time.time()
    test_duration = 5  # seconds
    
    print(f"⏱️ Recording for {test_duration} seconds...")
    
    while time.time() - start_time < test_duration:
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            # Basic processing
            if frame.shape[1] > 640:
                frame = cv2.resize(frame, (640, 480))
    
    cap.release()
    elapsed = time.time() - start_time
    fps = frame_count / elapsed
    
    print(f"✅ Baseline FPS: {fps:.1f}")
    return fps

def test_with_ai_moderator():
    """Test with actual AI moderator (brief test)"""
    print("\n🤖 Testing with AI Moderator...")
    
    try:
        from main import AIModerator
        
        # Create moderator in headless mode
        moderator = AIModerator(gui_mode=False)
        
        if not moderator.initialize_camera():
            print("❌ Could not initialize camera")
            return 0
        
        print("⏱️ Running AI processing test for 8 seconds...")
        
        frame_count = 0
        ai_count = 0
        start_time = time.time()
        test_duration = 8
        
        moderator.is_running = True
        
        while time.time() - start_time < test_duration and moderator.is_running:
            ret, frame = moderator.cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 4th frame with AI (similar to optimization)
            if frame_count % 4 == 0:
                processed_frame = moderator.process_frame(frame)
                ai_count += 1
            
            # Update FPS occasionally
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                print(f"   Current FPS: {current_fps:.1f}")
        
        moderator.cleanup()
        
        elapsed = time.time() - start_time
        total_fps = frame_count / elapsed
        ai_fps = ai_count / elapsed
        
        print(f"✅ AI Moderator Total FPS: {total_fps:.1f}")
        print(f"✅ AI Processing FPS: {ai_fps:.1f}")
        
        return total_fps
        
    except Exception as e:
        print(f"❌ Error testing AI moderator: {e}")
        return 0

def main():
    """Main test function"""
    print("🚀 AI Moderator FPS Test")
    print("=" * 40)
    print("Testing optimizations on your MacBook Pro i9...")
    
    # Test 1: Baseline
    baseline_fps = test_camera_only()
    
    # Test 2: With AI
    ai_fps = test_with_ai_moderator()
    
    # Results
    print("\n📊 RESULTS SUMMARY")
    print("=" * 30)
    print(f"Baseline FPS: {baseline_fps:.1f}")
    print(f"AI Moderator FPS: {ai_fps:.1f}")
    
    if ai_fps > 0:
        efficiency = (ai_fps / baseline_fps) * 100
        print(f"Efficiency: {efficiency:.1f}% of baseline")
        
        if ai_fps >= 20:
            print("🎉 EXCELLENT: >20 FPS - Great performance!")
        elif ai_fps >= 15:
            print("✅ GOOD: 15-20 FPS - Acceptable for real-time")
        elif ai_fps >= 10:
            print("⚠️ FAIR: 10-15 FPS - Usable but could be better")
        else:
            print("❌ POOR: <10 FPS - Needs more optimization")
    
    print("\n💡 The optimizations include:")
    print("   • AI processing every 4th frame instead of every frame")
    print("   • Cached face detection for 500ms intervals")
    print("   • Reduced GUI update frequency")
    print("   • Optimized PyAudio buffer sizes")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()