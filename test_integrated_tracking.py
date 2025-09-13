#!/usr/bin/env python3
"""
Test the integrated face detection + activity tracking system
Demonstrates enhanced face tracking with person identity management
"""

import cv2
import time
import signal
import sys
from integrated_tracker import IntegratedTracker

def signal_handler(sig, frame):
    print('\\n⏹️ Stopping test...')
    sys.exit(0)

def main():
    """Test integrated tracking with camera"""
    print("🎯 Enhanced Face Detection + Activity Tracking Test")
    print("="*60)
    print("📹 Controls:")
    print("   'q' - Quit")
    print("   'r' - Reset session") 
    print("   'e' - Export session data")
    print("   'h' - Show/hide help")
    print("   'ESC' - Quit")
    print("="*60)
    
    # Set up signal handler for clean exit
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize integrated tracker with person limit
    max_people = 6
    tracker = IntegratedTracker(max_people=max_people)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print(f"✅ Camera opened successfully")
    print(f"🎯 Tracking up to {max_people} people")
    
    frame_count = 0
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    show_help = True
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Could not read frame")
                break
            
            frame_count += 1
            fps_counter += 1
            
            # Calculate FPS
            current_time = time.time()
            if current_time - fps_start_time >= 1.0:
                current_fps = fps_counter / (current_time - fps_start_time)
                fps_counter = 0
                fps_start_time = current_time
            
            # Process frame with integrated tracker
            results = tracker.process_frame(frame)
            
            # Draw enhanced overlay
            display_frame = tracker.draw_overlay(frame)
            
            # Add test-specific information
            height, width = display_frame.shape[:2]
            
            # Add FPS counter
            cv2.putText(display_frame, f"FPS: {current_fps:.1f}", 
                       (width - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add frame counter
            cv2.putText(display_frame, f"Frame: {frame_count}", 
                       (width - 120, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add help text
            if show_help:
                help_lines = [
                    "Controls:",
                    "q/ESC - Quit",
                    "r - Reset",
                    "e - Export",
                    "h - Toggle help"
                ]
                for i, line in enumerate(help_lines):
                    cv2.putText(display_frame, line, (10, height - 120 + i*20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Show frame
            cv2.imshow('Enhanced Face Tracking Test', display_frame)
            
            # Print results every 60 frames (about 2 seconds)
            if frame_count % 60 == 0:
                summary = tracker.get_summary()
                people_present = tracker.get_present_people_list()
                
                print(f"\\n📊 Frame {frame_count} Summary:")
                print(f"   🎭 People present: {len(people_present)}/{summary['total_people_seen']}")
                print(f"   ⏱️  Duration: {summary['discussion_duration']:.1f}s")
                print(f"   🏆 Quality: {summary['discussion_quality']:.2f}")
                print(f"   ⚖️  Balance: {summary['turn_taking_balance']:.2f}")
                
                if people_present:
                    print("   👥 Active people:")
                    for person in people_present:
                        status = "🗣️ Speaking" if person['is_speaking'] else "👂 Listening"
                        print(f"      {person['name']}: {status} ({person['speaking_time']:.1f}s total)")
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('r'):  # Reset session
                print("\\n🔄 Resetting session...")
                tracker.reset_session()
                frame_count = 0
                print("✅ Session reset completed")
            elif key == ord('e'):  # Export session
                print("\\n💾 Exporting session...")
                filename = tracker.export_session()
                print(f"✅ Session exported to: {filename}")
            elif key == ord('h'):  # Toggle help
                show_help = not show_help
                print(f"📋 Help {'shown' if show_help else 'hidden'}")
    
    except KeyboardInterrupt:
        print("\\n⏹️ Test interrupted by user")
    
    except Exception as e:
        print(f"\\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\\n🧹 Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        
        # Final export and summary
        try:
            print("💾 Exporting final session data...")
            filename = tracker.export_session()
            
            # Print final summary
            summary = tracker.get_summary()
            people_list = tracker.get_present_people_list()
            all_people = tracker.face_tracker.get_all_people()
            
            print("\\n" + "="*60)
            print("📊 FINAL SESSION SUMMARY")
            print("="*60)
            print(f"⏱️  Total Duration: {summary['discussion_duration']:.1f} seconds")
            print(f"🖼️  Frames Processed: {summary['frames_processed']}")
            print(f"👥 People Seen: {len(all_people)} (max {max_people})")
            print(f"🎭 Currently Present: {len(people_list)}")
            print(f"🏆 Discussion Quality: {summary['discussion_quality']:.2f}")
            print(f"⚖️  Turn-taking Balance: {summary['turn_taking_balance']:.2f}")
            
            if all_people:
                print("\\n👥 All People Tracked:")
                for person_id, person in all_people.items():
                    status = "✅ Present" if person.is_present else "❌ Absent"
                    appearances = person.total_appearances
                    avg_conf = sum(person.confidence_history) / len(person.confidence_history) if person.confidence_history else 0
                    print(f"   {person.name}: {status} ({appearances} appearances, {avg_conf:.2f} confidence)")
            
            print(f"\\n💾 Final data saved to: {filename}")
            print("="*60)
            print("✅ Test completed successfully!")
        
        except Exception as e:
            print(f"❌ Error in cleanup: {e}")

if __name__ == "__main__":
    main()