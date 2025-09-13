"""
Headless AI Discussion Moderator - Main Application
Version without GUI for systems where tkinter is not available
"""

import cv2
import numpy as np
import time
import threading
import argparse
from datetime import datetime
from typing import Dict, List, Optional

# Import our modules
from face_tracker import FaceTracker, Face
from mouth_tracker import MouthTracker, SpeechActivity
from activity_tracker import ActivityTracker, ParticipantStats

class AIModeratorHeadless:
    """Headless AI Moderator application class"""
    
    def __init__(self, camera_index=0):
        """
        Initialize the AI Moderator
        
        Args:
            camera_index: Camera device index (default 0)
        """
        self.camera_index = camera_index
        
        # Initialize AI modules
        self.face_tracker = FaceTracker()
        self.mouth_tracker = MouthTracker()
        self.activity_tracker = ActivityTracker()
        
        # Initialize camera
        self.cap = None
        self.is_running = False
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Statistics display
        self.last_stats_time = time.time()
        self.stats_interval = 5.0  # Show stats every 5 seconds
        
    def initialize_camera(self) -> bool:
        """Initialize camera capture"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"ERROR: Could not open camera {self.camera_index}")
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print("Camera initialized successfully")
            return True
            
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the AI pipeline
        
        Args:
            frame: Input video frame
            
        Returns:
            Processed frame with annotations
        """
        # Detect faces
        faces = self.face_tracker.detect_faces(frame)
        
        # Analyze mouth movements for detected faces
        speech_activities = self.mouth_tracker.analyze_mouth_movement(faces)
        
        # Update activity tracking
        self.activity_tracker.update_activity(speech_activities)
        
        # Draw visualizations
        frame = self.face_tracker.draw_faces(frame, faces)
        frame = self.mouth_tracker.draw_mouth_analysis(frame, faces, speech_activities)
        
        # Add overlay information
        frame = self._add_overlay_info(frame, faces, speech_activities)
        
        return frame
    
    def _add_overlay_info(self, frame: np.ndarray, faces: List[Face], 
                         speech_activities: Dict[int, SpeechActivity]) -> np.ndarray:
        """Add overlay information to the frame"""
        height, width = frame.shape[:2]
        
        # Add FPS counter
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add participant count
        cv2.putText(frame, f"Participants: {len(faces)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add currently speaking count
        speaking_count = sum(1 for activity in speech_activities.values() if activity.is_speaking)
        cv2.putText(frame, f"Speaking: {speaking_count}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add participation summary (right side)
        activity_matrix = self.activity_tracker.get_activity_matrix()
        y_offset = 30
        
        for person_id, stats in activity_matrix.items():
            participation_level = stats['participation_level']
            speaking_percentage = stats['speaking_percentage']
            
            # Color based on participation level
            if participation_level == "dominant":
                color = (0, 0, 255)  # Red
            elif participation_level == "balanced":
                color = (0, 255, 0)  # Green
            elif participation_level == "quiet":
                color = (0, 255, 255)  # Yellow
            else:
                color = (128, 128, 128)  # Gray
            
            text = f"P{person_id}: {speaking_percentage:.1f}% ({participation_level})"
            cv2.putText(frame, text, (width - 350, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25
        
        return frame
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def print_statistics(self):
        """Print statistics to console"""
        current_time = time.time()
        
        if current_time - self.last_stats_time >= self.stats_interval:
            activity_matrix = self.activity_tracker.get_activity_matrix()
            
            print("\n" + "="*60)
            print(f"📊 DISCUSSION STATISTICS - {datetime.now().strftime('%H:%M:%S')}")
            print("="*60)
            
            if activity_matrix:
                print(f"👥 Total Participants: {len(activity_matrix)}")
                print(f"🎯 Currently Active: {sum(1 for stats in activity_matrix.values() if stats['is_active_now'])}")
                print()
                
                # Sort participants by speaking time
                sorted_participants = sorted(activity_matrix.items(), 
                                           key=lambda x: x[1]['speaking_percentage'], 
                                           reverse=True)
                
                print("📈 PARTICIPATION BREAKDOWN:")
                print("-" * 50)
                for person_id, stats in sorted_participants:
                    level = stats['participation_level']
                    percentage = stats['speaking_percentage']
                    speaking_time = stats['speaking_time']
                    
                    # Add emoji based on participation level
                    if level == "dominant":
                        emoji = "🔴"
                    elif level == "balanced":
                        emoji = "🟢"
                    elif level == "quiet":
                        emoji = "🟡"
                    else:
                        emoji = "⚪"
                    
                    print(f"{emoji} Person {person_id}: {percentage:.1f}% ({speaking_time:.1f}s) - {level.upper()}")
                
                # Show recommendations
                dominant_count = sum(1 for stats in activity_matrix.values() if stats['participation_level'] == 'dominant')
                quiet_count = sum(1 for stats in activity_matrix.values() if stats['participation_level'] == 'quiet')
                silent_count = sum(1 for stats in activity_matrix.values() if stats['participation_level'] == 'silent')
                
                print()
                print("💡 RECOMMENDATIONS:")
                print("-" * 30)
                
                if dominant_count > 0:
                    print("• Some participants are dominating - encourage turn-taking")
                if quiet_count > 1:
                    print("• Several participants are quiet - actively engage them")
                if silent_count > 0:
                    print("• Some participants haven't spoken - check if they need support")
                if dominant_count == 0 and quiet_count <= 1:
                    print("• Good participation balance! 🎉")
                
            else:
                print("👥 No participants detected yet")
            
            print("="*60)
            self.last_stats_time = current_time
    
    def run(self):
        """Run the headless application"""
        if not self.initialize_camera():
            return
        
        print("🤖 AI Discussion Moderator (Headless Mode)")
        print("="*50)
        print("📹 Camera feed will show in a window")
        print("📊 Statistics will be printed to console every 5 seconds")
        print("🔄 Press 'q' in the video window to quit")
        print("="*50)
        
        self.is_running = True
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Update FPS
                self.update_fps()
                
                # Print statistics periodically
                self.print_statistics()
                
                # Display frame
                cv2.imshow("AI Discussion Moderator - Live Feed", processed_frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
        
        except KeyboardInterrupt:
            print("\n🛑 Stopping application...")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n" + "="*60)
        print("📋 FINAL SESSION REPORT")
        print("="*60)
        
        activity_matrix = self.activity_tracker.get_activity_matrix()
        if activity_matrix:
            discussion_duration = time.time() - self.activity_tracker.discussion_start_time
            total_speaking_time = sum(stats['speaking_time'] for stats in activity_matrix.values())
            
            print(f"⏱️  Session Duration: {discussion_duration/60:.1f} minutes")
            print(f"🗣️  Total Speaking Time: {total_speaking_time/60:.1f} minutes")
            print(f"🤐 Silence Time: {(discussion_duration - total_speaking_time)/60:.1f} minutes")
            print()
            
            # Final participant summary
            sorted_participants = sorted(activity_matrix.items(), 
                                       key=lambda x: x[1]['speaking_percentage'], 
                                       reverse=True)
            
            print("🏆 FINAL RANKINGS:")
            for i, (person_id, stats) in enumerate(sorted_participants, 1):
                medal = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else f"{i}."
                print(f"{medal} Person {person_id}: {stats['speaking_percentage']:.1f}% - {stats['participation_level'].upper()}")
        
        print("="*60)
        print("✨ AI Discussion Moderator stopped")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AI Discussion Moderator (Headless)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    
    args = parser.parse_args()
    
    # Create moderator instance
    moderator = AIModeratorHeadless(camera_index=args.camera)
    
    try:
        moderator.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        moderator.cleanup()

if __name__ == "__main__":
    main()