#!/usr/bin/env python3
"""
Test different camera configurations to find what works
"""

import cv2
import time

def test_camera_config(config_name, settings_func):
    """Test a specific camera configuration"""
    print(f"\n🧪 Testing: {config_name}")
    print("-" * 40)
    
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Failed to open camera")
            return False
        
        # Apply settings
        if settings_func:
            settings_func(cap)
        
        # Test reading
        ret, frame = cap.read()
        
        if ret:
            print(f"✅ SUCCESS: Frame shape {frame.shape}")
            
            # Try reading a few more frames
            for i in range(3):
                ret, frame = cap.read()
                if not ret:
                    print(f"❌ Failed on frame {i+2}")
                    cap.release()
                    return False
            
            print("✅ Multiple frames read successfully")
            cap.release()
            return True
        else:
            print("❌ Failed to read first frame")
            cap.release()
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def main():
    print("🔧 Camera Configuration Testing")
    print("=" * 50)
    
    configs = [
        ("Default (no settings)", None),
        
        ("Basic resolution only", lambda cap: (
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640),
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        )),
        
        ("With FPS", lambda cap: (
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640),
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480),
            cap.set(cv2.CAP_PROP_FPS, 30)
        )),
        
        ("With buffer", lambda cap: (
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640),
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480),
            cap.set(cv2.CAP_PROP_FPS, 30),
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        )),
        
        ("With MJPEG codec", lambda cap: (
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640),
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480),
            cap.set(cv2.CAP_PROP_FPS, 30),
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1),
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        )),
        
        ("Lower resolution", lambda cap: (
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320),
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        )),
        
        ("Lower FPS", lambda cap: (
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640),
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480),
            cap.set(cv2.CAP_PROP_FPS, 15)
        )),
        
        ("No MJPEG codec", lambda cap: (
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640),
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480),
            cap.set(cv2.CAP_PROP_FPS, 30),
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        )),
    ]
    
    results = []
    
    for config_name, settings_func in configs:
        success = test_camera_config(config_name, settings_func)
        results.append((config_name, success))
        time.sleep(0.5)  # Brief pause between tests
    
    print("\n" + "=" * 50)
    print("📊 FINAL RESULTS:")
    print("=" * 50)
    
    working_configs = []
    for config_name, success in results:
        status = "✅ WORKS" if success else "❌ FAILS"
        print(f"  {config_name:<25} {status}")
        if success:
            working_configs.append(config_name)
    
    if working_configs:
        print(f"\n🎉 Working configurations found: {len(working_configs)}")
        print("💡 Recommendation: Use the simplest working config")
        print(f"   Best choice: {working_configs[0]}")
    else:
        print("\n⚠️  No working configurations found!")
        print("💡 Try checking camera permissions or use different camera index")

if __name__ == "__main__":
    main()