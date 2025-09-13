"""
Enhanced AI Moderator with Smooth OpenCV GUI
Uses direct OpenCV display for smooth, non-flickering video
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

class SmoothEnhancedModerator:
    """Enhanced AI Moderator with smooth OpenCV display"""
    
    def __init__(self, camera_index=0, max_people=8):
        """
        Initialize the Smooth Enhanced AI Moderator
        
        Args:
            camera_index: Camera device index (default 0)
            max_people: Maximum number of people to track (default 8)
        """
        self.camera_index = camera_index
        self.max_people = max_people
        
        # Initialize enhanced integrated tracker
        self.tracker = IntegratedTracker(max_people=max_people)
        self.analytics = DiscussionAnalytics()
        
        # Initialize camera
        self.cap = None
        self.is_running = False
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.last_fps_update = time.time()
        
        # Statistics tracking
        self.start_time = time.time()
        self.frame_count = 0
        self.speaking_events = 0
        
        # Window name
        self.window_name = "Enhanced AI Moderator - Smooth Display"
        
    def initialize_camera(self):
        """Initialize camera with optimized settings"""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise Exception(f"Cannot open camera {self.camera_index}")
        
        # Optimize camera settings for performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        
        print("✅ Camera initialized with optimized settings")
        
    def update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:  # Update every second
            elapsed = current_time - self.fps_start_time
            self.current_fps = self.fps_counter / elapsed if elapsed > 0 else 0
            self.fps_counter = 0
            self.fps_start_time = current_time
            self.last_fps_update = current_time
    
    def draw_enhanced_overlay(self, frame):
        """Draw enhanced information overlay without flickering"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay for stats
        overlay = frame.copy()
        
        # Top panel for main stats
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Bottom panel for controls
        cv2.rectangle(overlay, (0, height - 60), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Get tracking data
        people = self.tracker.get_present_people_list()
        speaking_people = []
        
        # Check if tracker has speech detection capability
        if hasattr(self.tracker.face_tracker, 'get_speaking_people'):
            speaking_people = self.tracker.face_tracker.get_speaking_people()
        
        # Count speaking events
        if speaking_people:
            self.speaking_events += 1
        
        # Main stats (top panel)
        stats_y = 25
        cv2.putText(frame, f"Enhanced AI Moderator", (10, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"People: {len(people)}", (10, stats_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Speaking: {len(speaking_people)}", (150, stats_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (280, stats_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Session stats (top right)
        elapsed_time = time.time() - self.start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        cv2.putText(frame, f"Session: {minutes:02d}:{seconds:02d}", (width - 200, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        speaking_rate = (self.speaking_events / self.frame_count * 100) if self.frame_count > 0 else 0
        cv2.putText(frame, f"Activity: {speaking_rate:.1f}%", (width - 200, stats_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)
        
        # Controls (bottom panel)
        controls_y = height - 35
        cv2.putText(frame, "Controls: [Q]uit | [R]eset | [S]witch Method | [E]xport", 
                   (10, controls_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.putText(frame, f"Method: {self.tracker.face_tracker.active_method.upper()}", 
                   (10, controls_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
        
        return frame
    
    def process_frame(self, frame):
        """Process a single frame with tracking and analytics"""
        # Process frame with integrated tracking
        tracking_results = self.tracker.process_frame(frame)
        
        # Draw the overlay from integrated tracker
        display_frame = self.tracker.draw_overlay(frame)
        
        # Add enhanced overlay
        display_frame = self.draw_enhanced_overlay(display_frame)
        
        return display_frame
    
    def handle_key_input(self, key):
        """Handle keyboard input"""
        if key == ord('q') or key == ord('Q'):
            return False  # Quit
        elif key == ord('r') or key == ord('R'):
            # Reset session
            self.tracker.reset_session()
            self.start_time = time.time()
            self.frame_count = 0
            self.speaking_events = 0
            print("🔄 Session reset")
        elif key == ord('s') or key == ord('S'):
            # Switch detection method
            current_method = self.tracker.face_tracker.active_method
            new_method = 'haar' if current_method == 'dnn' else 'dnn'
            self.tracker.face_tracker.set_detection_method(new_method)
            print(f"🔄 Switched to {self.tracker.face_tracker.active_method.upper()} method")
        elif key == ord('e') or key == ord('E'):
            # Export session data
            self.export_session_data()
        
        return True  # Continue
    
    def export_session_data(self):
        """Export current session data"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_export_{timestamp}.json"
            
            # Create export data
            export_data = {
                "session_info": {
                    "timestamp": timestamp,
                    "duration_seconds": time.time() - self.start_time,
                    "total_frames": self.frame_count,
                    "average_fps": self.current_fps
                },
                "people_tracked": len(self.tracker.get_all_people()),
                "speaking_events": self.speaking_events,
                "method_used": self.tracker.face_tracker.active_method
            }
            
            import json
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"📁 Session data exported to {filename}")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    def run(self):
        """Main processing loop"""
        try:
            self.initialize_camera()
            
            # Create window
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            
            print("🚀 Starting Enhanced AI Moderator with smooth display")
            print("=" * 60)
            print("📹 Camera initialized and ready")
            print("🎯 Enhanced face tracking active")
            print("🎤 Speech detection enabled")
            print("⚡ Optimized for smooth performance")
            print("=" * 60)
            
            self.is_running = True
            
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                # Process frame
                display_frame = self.process_frame(frame)
                
                # Update performance counters
                self.frame_count += 1
                self.fps_counter += 1
                self.update_fps()
                
                # Display frame (smooth, no flickering)
                cv2.imshow(self.window_name, display_frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Key was pressed
                    if not self.handle_key_input(key):
                        break
        
        except KeyboardInterrupt:
            print("\n⏹️ Interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Final stats
        elapsed_time = time.time() - self.start_time
        final_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        speaking_rate = (self.speaking_events / self.frame_count * 100) if self.frame_count > 0 else 0
        
        print("\n📊 Final Session Statistics:")
        print(f"   Duration: {elapsed_time:.1f} seconds")
        print(f"   Total frames: {self.frame_count}")
        print(f"   Average FPS: {final_fps:.1f}")
        print(f"   People tracked: {len(self.tracker.get_present_people_list())}")
        print(f"   Speaking events: {self.speaking_events}")
        print(f"   Activity rate: {speaking_rate:.1f}%")
        print(f"   Detection method: {self.tracker.face_tracker.active_method.upper()}")
        
        print("✅ Enhanced AI Moderator stopped")

def main():
    parser = argparse.ArgumentParser(description='Enhanced AI Moderator with Smooth Display')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--max-people', type=int, default=8, help='Maximum people to track (default: 8)')
    
    args = parser.parse_args()
    
    # Create and run the smooth enhanced moderator
    moderator = SmoothEnhancedModerator(
        camera_index=args.camera,
        max_people=args.max_people
    )
    
    moderator.run()

if __name__ == "__main__":
    main()