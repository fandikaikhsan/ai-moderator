#!/usr/bin/env python3
"""
Test script for AI Discussion Moderator
Tests basic functionality without GUI
"""

import cv2
import numpy as np
import time

def test_camera():
    """Test camera access"""
    print("Testing camera access...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot access camera")
        return False
    
    print("✅ Camera access successful")
    cap.release()
    return True

def test_face_detection():
    """Test OpenCV face detection"""
    print("Testing face detection...")
    
    try:
        # Test Haar cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Create a test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        # This won't detect anything in a black image, but tests if the cascade loads
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        print("✅ Haar cascade face detection loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Face detection failed: {e}")
        return False

def test_imports():
    """Test all module imports"""
    print("Testing module imports...")
    
    try:
        from face_tracker import FaceTracker
        print("✅ FaceTracker imported")
        
        from mouth_tracker import MouthTracker
        print("✅ MouthTracker imported")
        
        from activity_tracker import ActivityTracker
        print("✅ ActivityTracker imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def simple_camera_test():
    """Simple camera test with face detection"""
    print("Starting simple camera test...")
    print("Press 'q' to quit")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"Person", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Add info
            cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('AI Moderator - Simple Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Simple test completed")

def main():
    """Run all tests"""
    print("🤖 AI Discussion Moderator - Test Suite")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Camera
    if test_camera():
        tests_passed += 1
    
    # Test 2: Face detection
    if test_face_detection():
        tests_passed += 1
    
    # Test 3: Module imports
    if test_imports():
        tests_passed += 1
    
    print(f"\n📊 Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Running simple camera test...")
        simple_camera_test()
    else:
        print("❌ Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main()