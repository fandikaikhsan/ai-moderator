"""
Simple DNN/Haar Face Detection Demo with Speech Detection
Fast and reliable face tracking without YOLOv11 complexity
"""

import cv2
import argparse
import time
from enhanced_face_tracker import EnhancedFaceTracker

def main():
    parser = argparse.ArgumentParser(description='DNN/Haar Face Detection with Speech Analysis')
    parser.add_argument('--method', choices=['dnn', 'haar'], default='dnn', 
                       help='Detection method (default: dnn)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--no-gui', action='store_true', help='Run without GUI')
    
    args = parser.parse_args()
    
    print("🎯 Fast Face Detection with Speech Analysis")
    print("=" * 50)
    print(f"📹 Using camera: {args.camera}")
    print(f"🔧 Detection method: {args.method.upper()}")
    print("🎤 Speech detection: Enabled (simple algorithm)")
    print("⚡ Optimized for speed and reliability")
    print("=" * 50)
    
    # Initialize face tracker
    tracker = EnhancedFaceTracker(detection_method=args.method)
    
    if not tracker.active_method:
        print("❌ No detection method available")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"❌ Cannot open camera {args.camera}")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("🚀 Starting detection... (Press 'q' to quit, 's' to switch method)")
    
    # Performance tracking
    frame_count = 0
    start_time = time.time()
    last_stats_time = start_time
    speaking_detections = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Update tracking
            people = tracker.update_tracking(frame)
            
            # Count speaking people
            speaking_people = tracker.get_speaking_people()
            if speaking_people:
                speaking_detections += 1
            
            frame_count += 1
            
            # Display results
            if not args.no_gui:
                # Draw tracking info with speech indicators
                display_frame = tracker.draw_tracking_info_with_speech(frame)
                
                # Add performance info
                current_time = time.time()
                elapsed = current_time - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                cv2.putText(display_frame, f"Method: {tracker.active_method.upper()}", 
                           (10, display_frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                           (10, display_frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Speech detections: {speaking_detections}", 
                           (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Fast Face Detection with Speech', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Switch method
                    new_method = 'haar' if tracker.active_method == 'dnn' else 'dnn'
                    tracker.set_detection_method(new_method)
                    print(f"🔄 Switched to {tracker.active_method.upper()} method")
            else:
                # Console output mode
                current_time = time.time()
                if current_time - last_stats_time >= 2.0:  # Every 2 seconds
                    elapsed = current_time - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    speaking_rate = (speaking_detections / frame_count * 100) if frame_count > 0 else 0
                    
                    print(f"📊 Frame {frame_count}: 👥 People: {len(people)} | "
                          f"🗣️ Speaking: {len(speaking_people)} | "
                          f"⚡ FPS: {fps:.1f} | "
                          f"🎤 Speech rate: {speaking_rate:.1f}%")
                    
                    last_stats_time = current_time
    
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    
    finally:
        cap.release()
        if not args.no_gui:
            cv2.destroyAllWindows()
        
        # Final stats
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        speaking_rate = (speaking_detections / frame_count * 100) if frame_count > 0 else 0
        
        print("\n📊 Final Statistics:")
        print(f"   Method: {tracker.active_method.upper()}")
        print(f"   Total frames: {frame_count}")
        print(f"   Average FPS: {fps:.1f}")
        print(f"   Speech detections: {speaking_detections}")
        print(f"   Speech detection rate: {speaking_rate:.1f}%")
        print(f"   People tracked: {len(tracker.get_all_people())}")
        
        print("✅ Demo completed successfully!")

if __name__ == "__main__":
    main()