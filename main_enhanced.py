"""
AI Discussion Moderator - Enhanced Main Application
Integrates enhanced face tracking with activity monitoring
"""

import cv2
import numpy as np
import time
import threading
import argparse
from datetime import datetime
from typing import Dict, List, Optional

# Import our enhanced modules
from integrated_tracker import IntegratedTracker
from discussion_analytics import DiscussionAnalytics
from moderator_gui import ModeratorGUI

class EnhancedAIModerator:
    """Enhanced AI Moderator with improved face tracking and person management"""
    
    def __init__(self, camera_index=0, gui_mode=True, max_people=8):
        """
        Initialize the Enhanced AI Moderator
        
        Args:
            camera_index: Camera device index (default 0)
            gui_mode: Whether to run with GUI (True) or headless (False)
            max_people: Maximum number of people to track (default 8)
        """
        self.camera_index = camera_index
        self.gui_mode = gui_mode
        self.max_people = max_people
        
        # Initialize enhanced integrated tracker
        self.tracker = IntegratedTracker(max_people=max_people)
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
        
        print(f"🎯 Enhanced AI Moderator initialized (max {max_people} people)")
        
    def initialize_camera(self) -> bool:
        """Initialize camera capture with optimized settings"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"❌ ERROR: Could not open camera {self.camera_index}")
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Moderate resolution
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Moderate resolution
            self.cap.set(cv2.CAP_PROP_FPS, 30)            # High frame rate
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Minimal buffer for low latency
            
            # Additional optimizations for macOS
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            
            print("✅ Camera initialized successfully with optimized settings")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize camera: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the enhanced AI pipeline
        
        Args:
            frame: Input video frame
            
        Returns:
            Processed frame with annotations
        """
        try:
            # Process frame with integrated tracker
            results = self.tracker.process_frame(frame)
            
            # Draw enhanced overlay
            processed_frame = self.tracker.draw_overlay(frame)
            
            # Add additional information
            processed_frame = self._add_enhanced_overlay(processed_frame, results)
            
            return processed_frame
            
        except Exception as e:
            print(f"⚠️ Frame processing error: {e}")
            # Return frame with basic overlay on error
            return self._add_basic_overlay(frame)
    
    def _add_enhanced_overlay(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Add enhanced overlay information to the frame"""
        height, width = frame.shape[:2]
        
        # Add FPS counter (top-left)
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", 
                   (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add processing stats (bottom-left)
        stats_text = f"Frame: {results.get('frame_count', 0)} | Faces: {results.get('face_count', 0)}"
        cv2.putText(frame, stats_text, 
                   (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add session summary (top-right)
        try:
            summary = self.tracker.get_summary()
            summary_text = f"Duration: {summary['discussion_duration']:.1f}s | Quality: {summary['discussion_quality']:.2f}"
            text_size = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.putText(frame, summary_text, 
                       (width - text_size[0] - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception as e:
            pass  # Skip summary if there's an error
        
        return frame
    
    def _add_basic_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add basic overlay without expensive processing"""
        height, width = frame.shape[:2]
        
        # Add FPS counter
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", 
                   (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add basic info
        cv2.putText(frame, "Enhanced AI Moderator", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
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
        
        print("🚀 Starting Enhanced AI Moderator (headless mode)")
        print("📹 Press 'q' to quit, 'r' to reset session, 'e' to export")
        
        self.is_running = True
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read from camera")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Update FPS
                self.update_fps()
                
                # Display frame
                cv2.imshow("Enhanced AI Discussion Moderator", processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('r'):  # Reset session
                    print("🔄 Resetting session...")
                    self.tracker.reset_session()
                elif key == ord('e'):  # Export session
                    print("💾 Exporting session...")
                    filename = self.export_session_data()
                    if filename:
                        print(f"✅ Exported to: {filename}")
                
        except KeyboardInterrupt:
            print("\\n⏹️ Interrupted by user")
        
        finally:
            self.cleanup()
    
    def run_with_gui(self):
        """Run the application with GUI"""
        if not self.initialize_camera():
            return
        
        print("🚀 Starting Enhanced AI Moderator with GUI")
        
        # Create and configure GUI
        self.gui = ModeratorGUI()
        
        # Connect GUI callbacks
        self._setup_gui_integration()
        
        # Start the GUI
        self.gui.run()
    
    def _setup_gui_integration(self):
        """Setup integration between GUI and AI processing"""
        # Override GUI methods to integrate with enhanced AI processing
        original_start = self.gui.start_monitoring
        original_stop = self.gui.stop_monitoring
        
        def integrated_start():
            self.is_running = True
            # Start camera processing immediately when monitoring starts
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            original_start()
            print("✅ Enhanced AI processing started with GUI")
        
        def integrated_stop():
            self.is_running = False
            if self.processing_thread:
                self.processing_thread.join(timeout=1.0)
            original_stop()
            print("⏹️ Enhanced AI processing stopped")
        
        # Override GUI methods
        self.gui.start_monitoring = integrated_start
        self.gui.stop_monitoring = integrated_stop
        
        # Override data access methods with enhanced data
        self.gui.get_participant_data = self._get_gui_participant_data
        self.gui.get_analytics_data = self._get_gui_analytics_data
    
    def _get_gui_participant_data(self) -> Dict:
        """Get participant data formatted for GUI"""
        try:
            # Get people list from integrated tracker
            present_people = self.tracker.get_present_people_list()
            
            # Convert to format expected by GUI
            participant_data = {}
            for person in present_people:
                participant_data[person['id']] = {
                    'is_active_now': person['is_speaking'],
                    'speaking_percentage': (person['speaking_time'] / max(0.1, self.tracker.get_summary()['discussion_duration'])) * 100,
                    'participation_level': 'balanced',  # Could be enhanced based on speaking time
                    'name': person['name']
                }
            
            return participant_data
            
        except Exception as e:
            print(f"⚠️ Error getting participant data for GUI: {e}")
            return {}
    
    def _get_gui_analytics_data(self) -> Dict:
        """Get analytics data formatted for GUI"""
        try:
            # Get summary from integrated tracker
            summary = self.tracker.get_summary()
            
            # Format for GUI
            analytics_data = {
                'conversation_quality': {
                    'overall_score': summary['discussion_quality'],
                    'recommendations': [
                        f"Discussion has been running for {summary['discussion_duration']:.1f} seconds",
                        f"Currently tracking {summary['people_present']} people",
                        f"Total people seen: {summary['total_people_seen']}",
                        f"Turn-taking balance: {summary['turn_taking_balance']:.2f}"
                    ]
                },
                'turn_taking': {
                    'smooth_transitions': summary['turn_taking_balance'],
                    'overlaps': 0.1,  # Could be calculated from transcript
                    'interruptions': 0.05,  # Could be calculated from transcript
                    'silence_gaps': 0.1,
                    'transition_quality': summary['turn_taking_balance']
                }
            }
            
            return analytics_data
            
        except Exception as e:
            print(f"⚠️ Error getting analytics data for GUI: {e}")
            return {'conversation_quality': {'overall_score': 0, 'recommendations': []}}
    
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
                    print("❌ Failed to read from camera in GUI mode")
                    break
                
                # Skip frames for performance - process every 2nd frame
                frame_skip_counter += 1
                process_ai = (frame_skip_counter % 2 == 0)
                
                if process_ai:
                    # Process frame with enhanced AI
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
                print(f"⚠️ Processing loop error: {e}")
                time.sleep(0.1)  # Brief pause on error
    
    def cleanup(self):
        """Cleanup resources"""
        print("🧹 Cleaning up resources...")
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("✅ Enhanced AI Moderator stopped")
    
    def export_session_data(self, filename: Optional[str] = None) -> str:
        """Export current session data with enhanced tracking information"""
        try:
            filename = self.tracker.export_session(filename)
            return filename
        except Exception as e:
            print(f"❌ Failed to export session data: {e}")
            return ""
    
    def generate_report(self) -> Dict:
        """Generate a comprehensive discussion report with enhanced data"""
        try:
            # Get summary from integrated tracker
            summary = self.tracker.get_summary()
            present_people = self.tracker.get_present_people_list()
            
            # Calculate enhanced metrics
            total_participants = len(present_people)
            
            if total_participants > 0:
                # Find most and least active participants
                sorted_people = sorted(present_people, key=lambda x: x['speaking_time'], reverse=True)
                most_active = sorted_people[0] if sorted_people else None
                least_active = sorted_people[-1] if sorted_people else None
            else:
                most_active = least_active = None
            
            report = {
                'summary': {
                    'total_participants': total_participants,
                    'total_people_seen': summary['total_people_seen'],
                    'most_active_participant': most_active,
                    'least_active_participant': least_active,
                    'participation_balance': summary['turn_taking_balance'],
                    'discussion_quality': summary['discussion_quality'],
                    'discussion_duration': summary['discussion_duration'],
                    'frames_processed': summary['frames_processed']
                },
                'participants': {person['id']: person for person in present_people},
                'face_tracking': {
                    'max_people_limit': self.max_people,
                    'people_present': summary['people_present'],
                    'total_tracked': summary['total_people_seen']
                },
                'recommendations': [
                    f"Discussion ran for {summary['discussion_duration']:.1f} seconds",
                    f"Tracked {summary['total_people_seen']} different people",
                    f"Discussion quality score: {summary['discussion_quality']:.2f}",
                    f"Turn-taking balance: {summary['turn_taking_balance']:.2f}"
                ]
            }
            
            return report
            
        except Exception as e:
            print(f"⚠️ Error generating report: {e}")
            return {'summary': {}, 'participants': {}, 'recommendations': []}

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Enhanced AI Discussion Moderator")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI")
    parser.add_argument("--export", type=str, help="Export session data to file")
    parser.add_argument("--max-people", type=int, default=8, help="Maximum number of people to track (default: 8)")
    
    args = parser.parse_args()
    
    # Create enhanced moderator instance
    moderator = EnhancedAIModerator(
        camera_index=args.camera, 
        gui_mode=not args.no_gui,
        max_people=args.max_people
    )
    
    try:
        if args.no_gui:
            moderator.run_headless()
        else:
            moderator.run_with_gui()
            
    except KeyboardInterrupt:
        print("\\n⏹️ Shutting down...")
        
    finally:
        # Export data if requested
        if args.export:
            filename = moderator.export_session_data(args.export)
            if filename:
                print(f"✅ Final export saved to: {filename}")
        
        moderator.cleanup()

if __name__ == "__main__":
    main()