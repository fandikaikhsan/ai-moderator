#!/usr/bin/env python3
"""
Test script to verify enhanced face tracking capabilities
"""

import cv2
import time
import numpy as np
from face_tracker import FaceTracker

def test_face_tracking():
    """Test the enhanced face tracking system"""
    print("🎯 Testing Enhanced Face Tracking System")
    print("=" * 50)
    
    # Initialize face tracker
    face_tracker = FaceTracker()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    print("\n📹 Starting face tracking test...")
    print("📋 Instructions:")
    print("• Move around in front of the camera")
    print("• Look away and look back")
    print("• Cover your face temporarily")
    print("• Watch how the person ID stays consistent")
    print("• Press 'q' to quit")
    print("• Press 's' to show statistics")
    print()
    
    frame_count = 0
    face_id_history = []
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            frame_count += 1
            
            # Detect and track faces
            faces = face_tracker.detect_faces(frame)
            
            # Draw faces with enhanced tracking info
            frame = face_tracker.draw_faces(frame, faces)
            
            # Record face IDs for analysis
            current_ids = [face.id for face in faces]
            face_id_history.append(current_ids)
            
            # Add performance info
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Add test info overlay
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Tracking stability info
            if face_id_history:
                recent_ids = [ids for ids in face_id_history[-30:] if ids]  # Last 30 frames with faces
                if recent_ids:
                    all_ids = set()
                    for ids in recent_ids:
                        all_ids.update(ids)
                    
                    stability_text = f"IDs seen: {sorted(list(all_ids))}"
                    cv2.putText(frame, stability_text, (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Instructions
            cv2.putText(frame, "Press 'q' to quit, 's' for stats", (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow("Enhanced Face Tracking Test", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('s'):  # Show statistics
                print_tracking_statistics(face_id_history, elapsed)
    
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        print("\n" + "=" * 50)
        print("📊 FINAL TRACKING ANALYSIS")
        print("=" * 50)
        print_tracking_statistics(face_id_history, time.time() - start_time)

def print_tracking_statistics(face_id_history, elapsed_time):
    """Print detailed tracking statistics"""
    if not face_id_history:
        print("No tracking data available")
        return
    
    # Calculate statistics
    frames_with_faces = len([ids for ids in face_id_history if ids])
    total_frames = len(face_id_history)
    
    # Count unique IDs
    all_ids = set()
    for ids in face_id_history:
        all_ids.update(ids)
    
    # ID stability analysis
    id_changes = 0
    prev_ids = set()
    for ids in face_id_history:
        current_ids = set(ids)
        if prev_ids and current_ids and current_ids != prev_ids:
            id_changes += 1
        prev_ids = current_ids
    
    # Calculate tracking quality
    detection_rate = (frames_with_faces / max(total_frames, 1)) * 100
    stability_score = max(0, 100 - (id_changes / max(frames_with_faces, 1)) * 100)
    
    print(f"⏱️  Test Duration: {elapsed_time:.1f} seconds")
    print(f"📹 Total Frames: {total_frames}")
    print(f"👥 Frames with Faces: {frames_with_faces}")
    print(f"🆔 Unique IDs Created: {len(all_ids)}")
    print(f"🔄 ID Changes: {id_changes}")
    print(f"📊 Detection Rate: {detection_rate:.1f}%")
    print(f"🎯 Tracking Stability: {stability_score:.1f}%")
    print()
    
    if len(all_ids) <= 2:
        print("✅ EXCELLENT: Consistent person identification!")
    elif len(all_ids) <= 3:
        print("🟡 GOOD: Minor ID fluctuations")
    else:
        print("🔴 NEEDS IMPROVEMENT: Too many ID changes")
    
    print()
    print("💡 TRACKING QUALITY INDICATORS:")
    print(f"   📈 Stability Score > 80%: {'✅' if stability_score > 80 else '❌'}")
    print(f"   🎯 Low ID Count (≤2): {'✅' if len(all_ids) <= 2 else '❌'}")
    print(f"   📊 High Detection Rate: {'✅' if detection_rate > 90 else '❌'}")

if __name__ == "__main__":
    test_face_tracking()