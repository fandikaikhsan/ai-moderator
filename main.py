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
from speech_transcriber import SpeechTranscriber
from pyaudio_speech_transcriber import PyAudioSpeechTranscriber
from discussion_summarizer import DiscussionSummarizer

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
        
        # Initialize speech and summarization modules
        self.speech_transcriber = PyAudioSpeechTranscriber()
        self.discussion_summarizer = DiscussionSummarizer()
        
        # Initialize camera
        self.cap = None
        self.is_running = False
        
        # GUI components
        self.gui = None
        self.processing_thread = None
        
        # Speech transcription state
        self.transcription_active = False
        
        # Performance tracking - enhanced
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.frame_count = 0
        self.ai_processing_count = 0
        self.performance_stats = {
            'total_frames': 0,
            'ai_frames': 0,
            'avg_fps': 0,
            'ai_fps': 0
        }
        
    def initialize_camera(self) -> bool:
        """Initialize camera capture with default settings (works best on macOS)"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"ERROR: Could not open camera {self.camera_index}")
                return False
            
            # Use default camera settings - setting properties can cause issues on macOS
            # The camera will use its native resolution and settings for best compatibility
            print("Camera initialized successfully with default settings")
            return True
            
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the AI pipeline - optimized for performance
        
        Args:
            frame: Input video frame
            
        Returns:
            Processed frame with annotations
        """
        # Use existing faces if we have recent ones (for performance)
        if not hasattr(self, '_last_faces_time'):
            self._last_faces_time = 0
            self._cached_faces = []
        
        current_time = time.time()
        
        # Only detect faces every 500ms for better performance
        if current_time - self._last_faces_time > 0.5 or not self._cached_faces:
            faces = self.face_tracker.detect_faces(frame)
            self._cached_faces = faces
            self._last_faces_time = current_time
        else:
            faces = self._cached_faces
        
        self.current_faces = faces  # Store for GUI access
        
        # Analyze mouth movements for detected faces (less frequently)
        speech_activities = {}
        if faces and (current_time - getattr(self, '_last_mouth_analysis', 0)) > 0.3:
            speech_activities = self.mouth_tracker.analyze_mouth_movement(faces)
            self._last_mouth_analysis = current_time
            
            # Update activity tracking
            self.activity_tracker.update_activity(speech_activities)
        
        # Process speech transcription if active (throttled)
        if self.transcription_active and (current_time - getattr(self, '_last_transcription_check', 0)) > 0.5:
            active_face_ids = [face.id for face in faces]
            new_transcriptions = self.speech_transcriber.process_audio_queue(active_face_ids)
            
            # Update GUI with new transcriptions
            if self.gui and new_transcriptions:
                for participant_id, text in new_transcriptions.items():
                    self.gui.update_transcription(participant_id, text)
            
            self._last_transcription_check = current_time
        
        # Draw visualizations (optimized)
        frame = self.face_tracker.draw_faces(frame, faces)
        if speech_activities:
            frame = self.mouth_tracker.draw_mouth_analysis(frame, faces, speech_activities)
        
        # Add overlay information
        frame = self._add_overlay_info(frame, faces, speech_activities)
        
        return frame
        
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
        """Update FPS counter with enhanced performance tracking"""
        self.fps_counter += 1
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            elapsed = current_time - self.fps_start_time
            self.current_fps = self.fps_counter / elapsed
            
            # Update performance stats
            self.performance_stats['total_frames'] = self.frame_count
            self.performance_stats['ai_frames'] = self.ai_processing_count
            self.performance_stats['avg_fps'] = self.current_fps
            self.performance_stats['ai_fps'] = self.ai_processing_count / elapsed if elapsed > 0 else 0
            
            # Print performance info every 5 seconds
            if self.frame_count > 0 and self.current_fps > 0 and self.frame_count % max(1, int(5 * self.current_fps)) == 0:
                print(f"Performance: {self.current_fps:.1f} FPS | AI: {self.performance_stats['ai_fps']:.1f} FPS | "
                      f"AI Ratio: {(self.ai_processing_count/max(1,self.frame_count)*100):.1f}%")
            
            self.fps_counter = 0
            self.ai_processing_count = 0
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
        
        # Setup transcription callbacks
        self.gui.set_transcription_callback(self._handle_transcription_control)
        self.gui.set_summary_callback(self._generate_discussion_summary)
        self.gui.set_microphone_callback(self._handle_microphone_change)
        
        # Store reference to current faces
        self.current_faces = []
    
    def _handle_transcription_control(self, action: str):
        """Handle transcription control commands from GUI"""
        if action == 'start':
            self.transcription_active = True
            self.speech_transcriber.start_listening()
            print("Speech transcription started")
            
        elif action == 'stop':
            self.transcription_active = False
            self.speech_transcriber.stop_listening()
            print("Speech transcription stopped")
            
        elif action == 'clear':
            self.speech_transcriber.clear_transcriptions()
            print("Transcription data cleared")
    
    def _generate_discussion_summary(self):
        """Generate and display discussion summary"""
        try:
            # Get full transcript and stats
            full_transcript = self.speech_transcriber.get_full_transcript()
            discussion_stats = self.speech_transcriber.get_discussion_stats()
            
            if not full_transcript.strip():
                # No transcript available
                summary_data = {
                    'summary': "No discussion content available to summarize.",
                    'key_points': ["No discussion recorded"],
                    'participants_contribution': {"Info": "No participants detected"},
                    'action_items': ["No action items identified"],
                    'sentiment': "Neutral",
                    'duration_minutes': 0
                }
            else:
                # Generate summary using Ollama
                summary = self.discussion_summarizer.generate_summary(full_transcript, discussion_stats)
                summary_data = {
                    'summary': summary.summary,
                    'key_points': summary.key_points,
                    'participants_contribution': summary.participants_contribution,
                    'action_items': summary.action_items,
                    'sentiment': summary.sentiment,
                    'duration_minutes': summary.duration_minutes
                }
            
            # Display in GUI
            if self.gui:
                self.gui.display_summary(summary_data)
                
            print("Discussion summary generated")
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            # Show error in GUI
            if self.gui:
                error_summary = {
                    'summary': f"Failed to generate summary: {e}",
                    'key_points': ["Summary generation failed"],
                    'participants_contribution': {"Error": str(e)},
                    'action_items': ["Check Ollama service and model availability"],
                    'sentiment': "Unknown",
                    'duration_minutes': 0
                }
                self.gui.display_summary(error_summary)
    
    def _handle_microphone_change(self, microphone_index: int):
        """Handle microphone selection change from GUI"""
        try:
            success = self.speech_transcriber.change_microphone(microphone_index)
            if success:
                print(f"Successfully changed to microphone index {microphone_index}")
            else:
                print(f"Failed to change to microphone index {microphone_index}")
        except Exception as e:
            print(f"Error changing microphone: {e}")
    
    def _processing_loop(self):
        """Main processing loop for GUI mode - optimized for performance"""
        frame_skip_counter = 0
        ai_skip_counter = 0
        fps_target = 30  # Increased target FPS
        target_interval = 1.0 / fps_target
        last_frame_time = time.time()
        last_ai_time = time.time()
        
        # Performance optimizations
        ai_processing_interval = 0.2  # Process AI every 200ms (5 FPS for AI)
        gui_update_interval = 1.0 / 25  # Update GUI at 25 FPS max
        
        while self.is_running:
            try:
                current_time = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                # Reduce frame size for faster processing
                height, width = frame.shape[:2]
                if width > 640:
                    scale_factor = 640 / width
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                frame_skip_counter += 1
                ai_skip_counter += 1
                
                # Process AI much less frequently (every 200ms instead of every frame)
                time_since_ai = current_time - last_ai_time
                should_process_ai = time_since_ai >= ai_processing_interval
                
                if should_process_ai:
                    # Process frame with AI (expensive operations) - much less frequent
                    processed_frame = self.process_frame(frame)
                    last_ai_time = current_time
                    ai_skip_counter = 0
                    self.ai_processing_count += 1  # Track AI processing
                else:
                    # Just add basic overlay without AI processing
                    processed_frame = self._add_basic_overlay(frame)
                
                # Update FPS counter less frequently
                if frame_skip_counter % 5 == 0:
                    self.update_fps()
                
                # Update GUI with processed frame (limit frequency)
                time_since_last = current_time - last_frame_time
                if time_since_last >= gui_update_interval:
                    if self.gui and hasattr(self.gui, 'update_video_frame'):
                        # Use after instead of after_idle for more predictable timing
                        self.gui.root.after(1, lambda f=processed_frame.copy(): self.gui.update_video_frame(f))
                    last_frame_time = current_time
                
                # Minimal sleep to prevent CPU overload
                time.sleep(0.001)  # 1ms sleep
                
            except Exception as e:
                print(f"Error in processing loop: {e}")
                break
        
        print("AI Moderator stopped")
    
    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        
        # Stop transcription if active
        if self.transcription_active:
            self.speech_transcriber.stop_listening()
            self.transcription_active = False
        
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
