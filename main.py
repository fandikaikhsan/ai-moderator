"""
AI Discussion Moderator - Main Application
Integrates computer vision, speech analysis, and real-time monitoring
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
from discussion_analytics import DiscussionAnalytics
from moderator_gui import ModeratorGUI

class AIModerator:
    """Main AI Moderator application class"""
    
    def __init__(self, camera_index=0, gui_mode=True):
        """
        Initialize the AI Moderator
        
        Args:
            camera_index: Camera device index (default 0)
            gui_mode: Whether to run with GUI (True) or headless (False)
        """
        self.camera_index = camera_index
        self.gui_mode = gui_mode
        
        # Initialize AI modules
        self.face_tracker = FaceTracker()
        self.mouth_tracker = MouthTracker()
        self.activity_tracker = ActivityTracker()
        self.analytics = DiscussionAnalytics()
        
        # Initialize camera
        self.cap = None
        self.is_running = False
        
        # GUI components
        self.gui = None
        self.processing_thread = None
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def initialize_camera(self) -> bool:
        """Initialize camera capture with optimized settings"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"ERROR: Could not open camera {self.camera_index}")
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Moderate resolution
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Moderate resolution
            self.cap.set(cv2.CAP_PROP_FPS, 30)            # High frame rate
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Minimal buffer for low latency
            
            # Additional optimizations for macOS
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            
            print("Camera initialized successfully with optimized settings")
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
        self.current_faces = faces  # Store for GUI access
        
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
            cv2.putText(frame, text, (width - 300, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25
        
        return frame
    
    def _add_basic_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add basic overlay without expensive AI processing"""
        height, width = frame.shape[:2]
        
        # Add FPS counter
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add participant count from last known data
        participant_count = len(self.activity_tracker.get_activity_matrix())
        cv2.putText(frame, f"Participants: {participant_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def run_headless(self):
        """Run the application without GUI"""
        if not self.initialize_camera():
            return
        
        print("Starting AI Moderator (headless mode)")
        print("Press 'q' to quit")
        
        self.is_running = True
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Update FPS
            self.update_fps()
            
            # Display frame
            cv2.imshow("AI Discussion Moderator", processed_frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
        
        self.cleanup()
    
    def run_with_gui(self):
        """Run the application with GUI"""
        if not self.initialize_camera():
            return
        
        print("Starting AI Moderator with GUI")
        
        # Create and configure GUI
        self.gui = ModeratorGUI()
        
        # Connect GUI callbacks
        self._setup_gui_integration()
        
        # Start the GUI
        self.gui.run()
    
    def _setup_gui_integration(self):
        """Setup integration between GUI and AI processing"""
        # Override GUI methods to integrate with AI processing
        original_start = self.gui.start_monitoring
        original_stop = self.gui.stop_monitoring
        
        def integrated_start():
            self.is_running = True
            # Start camera processing immediately when monitoring starts
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            original_start()
            print("AI processing started with GUI")
        
        def integrated_stop():
            self.is_running = False
            if self.processing_thread:
                self.processing_thread.join(timeout=1.0)
            original_stop()
            print("AI processing stopped")
        
        # Override GUI methods
        self.gui.start_monitoring = integrated_start
        self.gui.stop_monitoring = integrated_stop
        
        # Override data access methods
        self.gui.get_participant_data = lambda: self.activity_tracker.get_activity_matrix()
        self.gui.get_analytics_data = lambda: self.analytics.analyze_discussion_flow(self.activity_tracker)
        
        # Store reference to current faces
        self.current_faces = []
    
    def _processing_loop(self):
        """Main processing loop for GUI mode - optimized for performance"""
        frame_skip_counter = 0
        fps_target = 15  # Target 15 FPS for good performance
        target_interval = 1.0 / fps_target
        last_frame_time = time.time()
        
        while self.is_running:
            try:
                current_time = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                # Skip frames for performance - process every 2nd frame for face detection
                frame_skip_counter += 1
                process_ai = (frame_skip_counter % 2 == 0)
                
                if process_ai:
                    # Process frame with AI (expensive operations)
                    processed_frame = self.process_frame(frame)
                else:
                    # Just add basic overlay without AI processing
                    processed_frame = self._add_basic_overlay(frame)
                
                # Update FPS counter
                self.update_fps()
                
                # Update GUI with processed frame (limit frequency)
                time_since_last = current_time - last_frame_time
                if time_since_last >= target_interval:
                    if self.gui and hasattr(self.gui, 'update_video_frame'):
                        # Use after_idle for better performance
                        self.gui.root.after_idle(lambda f=processed_frame: self.gui.update_video_frame(f))
                    last_frame_time = current_time
                
                # Adaptive sleep to maintain target FPS
                elapsed = time.time() - current_time
                sleep_time = max(0, target_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                print(f"Processing error: {e}")
                break
    
    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("AI Moderator stopped")
    
    def export_session_data(self, filename: Optional[str] = None) -> str:
        """Export current session data"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"discussion_session_{timestamp}.json"
        
        # Get session data from activity tracker
        session_data = self.activity_tracker.export_session_data()
        
        # Add analytics data
        analytics_data = self.analytics.analyze_discussion_flow(self.activity_tracker)
        session_data['analytics'] = analytics_data
        
        # Save to file
        import json
        try:
            with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            print(f"Session data exported to {filename}")
            return filename
        except Exception as e:
            print(f"Failed to export session data: {e}")
            return ""
    
    def generate_report(self) -> Dict:
        """Generate a comprehensive discussion report"""
        # Get all data
        activity_matrix = self.activity_tracker.get_activity_matrix()
        analytics_data = self.analytics.analyze_discussion_flow(self.activity_tracker)
        discussion_map = self.analytics.generate_discussion_map(self.activity_tracker)
        
        # Generate summary
        total_participants = len(activity_matrix)
        
        if total_participants > 0:
            # Find most and least active participants
            sorted_participants = sorted(activity_matrix.items(), 
                                       key=lambda x: x[1]['speaking_percentage'], 
                                       reverse=True)
            
            most_active = sorted_participants[0] if sorted_participants else None
            least_active = sorted_participants[-1] if sorted_participants else None
            
            # Calculate balance metrics
            speaking_percentages = [stats['speaking_percentage'] for stats in activity_matrix.values()]
            balance_score = 1.0 - (np.std(speaking_percentages) / np.mean(speaking_percentages)) if np.mean(speaking_percentages) > 0 else 0
        else:
            most_active = least_active = None
            balance_score = 0
        
        report = {
            'summary': {
                'total_participants': total_participants,
                'most_active_participant': most_active,
                'least_active_participant': least_active,
                'participation_balance': balance_score,
                'discussion_quality': analytics_data.get('conversation_quality', {}).get('overall_score', 0)
            },
            'participants': activity_matrix,
            'analytics': analytics_data,
            'discussion_map': discussion_map,
            'recommendations': analytics_data.get('conversation_quality', {}).get('recommendations', [])
        }
        
        return report

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AI Discussion Moderator")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI")
    parser.add_argument("--export", type=str, help="Export session data to file")
    
    args = parser.parse_args()
    
    # Create moderator instance
    moderator = AIModerator(camera_index=args.camera, gui_mode=not args.no_gui)
    
    try:
        if args.no_gui:
            moderator.run_headless()
        else:
            moderator.run_with_gui()
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        
    finally:
        # Export data if requested
        if args.export:
            moderator.export_session_data(args.export)
        
        moderator.cleanup()

if __name__ == "__main__":
    main()
