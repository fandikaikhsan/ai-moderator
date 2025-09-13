"""
Integrated Face Detection + Activity Tracking
Combines face tracking with speaking activity analysis
"""

import cv2
import time
import numpy as np
from typing import Dict, Optional, List
from enhanced_face_tracker import EnhancedFaceTracker, Person
from activity_tracker import ActivityTracker
from mouth_tracker import SpeechActivity
import json
from datetime import datetime

class IntegratedTracker:
    """Combines face detection with activity tracking"""
    
    def __init__(self, max_people: int = 8):
        """Initialize integrated tracking system"""
        self.face_tracker = EnhancedFaceTracker(max_people=max_people)
        self.activity_tracker = ActivityTracker()
        self.mouth_tracker = None  # Will be initialized when needed
        
        # Load mouth tracker if available
        try:
            from mouth_tracker import MouthTracker
            self.mouth_tracker = MouthTracker()
            print("✅ MouthTracker loaded successfully")
        except ImportError:
            print("⚠️ MouthTracker not available - using simulated speech detection")
        
        # Tracking state
        self.last_process_time = time.time()
        self.frame_count = 0
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame for face detection and activity tracking
        
        Args:
            frame: Input video frame
            
        Returns:
            Dictionary with tracking results
        """
        self.frame_count += 1
        current_time = time.time()
        
        # Update face tracking
        present_people = self.face_tracker.update_tracking(frame)
        
        # Generate speech activities for present people
        speech_activities = {}
        
        for person_id, person in present_people.items():
            if person.current_bbox is not None:
                # Extract face region for mouth analysis
                x, y, w, h = person.current_bbox
                
                # Ensure bbox is within frame bounds
                frame_height, frame_width = frame.shape[:2]
                x = max(0, min(x, frame_width - 1))
                y = max(0, min(y, frame_height - 1))
                w = min(w, frame_width - x)
                h = min(h, frame_height - y)
                
                if w > 0 and h > 0:
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Detect speech activity
                    if face_roi.size > 0:
                        speech_activity = self._detect_speech_activity(person, face_roi, frame)
                        speech_activities[person_id] = speech_activity
        
        # Update activity tracking
        if speech_activities:
            self.activity_tracker.update_activity(speech_activities)
        
        self.last_process_time = current_time
        
        return {
            'present_people': present_people,
            'speech_activities': speech_activities,
            'face_count': len(present_people),
            'total_tracked': len(self.face_tracker.get_all_people()),
            'frame_count': self.frame_count
        }
    
    def _detect_speech_activity(self, person: Person, face_roi: np.ndarray, full_frame: np.ndarray) -> SpeechActivity:
        """
        Detect speech activity for a person
        
        Args:
            person: Person being analyzed
            face_roi: Face region of interest
            full_frame: Full video frame
            
        Returns:
            SpeechActivity object
        """
        if self.mouth_tracker is not None:
            # Use real mouth tracking
            try:
                # Extract mouth region from face
                x, y, w, h = person.current_bbox
                mouth_landmarks = self.face_tracker._extract_mouth_region(full_frame, x, y, w, h)
                
                # Create a mock Face object for compatibility with MouthTracker
                from face_tracker import Face
                mock_face = Face(
                    id=person.person_id,
                    bbox=person.current_bbox,
                    landmarks=mouth_landmarks,
                    confidence=np.mean(list(person.confidence_history)) if person.confidence_history else 0.8,
                    last_seen=person.last_seen,
                    center=person.center_history[-1] if person.center_history else (0, 0)
                )
                
                # Detect speech using mouth tracker (correct method name)
                current_activities = self.mouth_tracker.analyze_mouth_movement([mock_face])
                
                if person.person_id in current_activities:
                    return current_activities[person.person_id]
                else:
                    # Return default inactive speech activity
                    return SpeechActivity(
                        person_id=person.person_id,
                        is_speaking=False,
                        speaking_intensity=0.0,
                        mouth_aspect_ratio=0.1,
                        lip_distance=5.0,
                        speaking_duration=0.0,
                        last_speech_time=time.time()
                    )
            
            except Exception as e:
                print(f"Error in mouth tracking for person {person.person_id}: {e}")
                # Fall through to simulated detection
        
        # Use simulated speech detection
        speaking_intensity = self._simulate_speech_detection(person, face_roi)
        is_speaking = speaking_intensity > 0.3
        
        return SpeechActivity(
            person_id=person.person_id,
            is_speaking=is_speaking,
            speaking_intensity=speaking_intensity,
            mouth_aspect_ratio=0.3 if is_speaking else 0.1,
            lip_distance=12.0 if is_speaking else 8.0,
            speaking_duration=1.0 if is_speaking else 0.0,
            last_speech_time=time.time() if is_speaking else time.time() - 1.0
        )
    
    def _simulate_speech_detection(self, person: Person, face_roi: np.ndarray) -> float:
        """
        Simulate speech detection when mouth tracker is not available
        
        Args:
            person: Person being analyzed
            face_roi: Face region of interest
            
        Returns:
            Simulated speaking intensity (0-1)
        """
        if face_roi.size == 0:
            return 0.0
        
        try:
            # Use face size as activity indicator
            if person.current_bbox is not None:
                _, _, w, h = person.current_bbox
                face_size_factor = min(1.0, (w * h) / (80 * 80))
            else:
                face_size_factor = 0.0
            
            # Use confidence as activity indicator
            confidence_factor = 0.0
            if person.confidence_history and len(person.confidence_history) > 0:
                confidence_factor = np.mean(list(person.confidence_history))
            
            # Add some randomness to simulate natural speaking patterns
            # Make it more realistic by having longer speaking/silence periods
            random_base = np.random.random()
            
            # Create more realistic speech patterns (bursts of activity)
            time_factor = (time.time() * 0.5) % 10  # Slow cycle
            speech_probability = 0.3 + 0.2 * np.sin(time_factor + person.person_id)  # Different phase per person
            
            if random_base > speech_probability:
                activity_level = 0.1 + random_base * 0.2  # Low activity
            else:
                activity_level = 0.4 + random_base * 0.5  # Speaking activity
            
            # Combine factors
            speaking_intensity = (
                face_size_factor * 0.4 + 
                confidence_factor * 0.3 + 
                activity_level * 0.3
            )
            
            return min(1.0, max(0.0, speaking_intensity))
        
        except Exception as e:
            print(f"Error in speech simulation: {e}")
            return 0.0
    
    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw complete tracking overlay on frame"""
        # Draw face tracking info
        output_frame = self.face_tracker.draw_tracking_info(frame)
        
        # Add activity information
        try:
            # Get current participants from activity tracker
            participants = self.activity_tracker.participants
            
            y_offset = 60
            for person_id, participant in participants.items():
                # Check if currently speaking based on active session
                is_currently_speaking = participant.current_session_start is not None
                
                if is_currently_speaking:
                    # Show currently speaking
                    text = f"🗣️ Person {person_id}: Speaking"
                    color = (0, 255, 255)  # Yellow for speaking
                else:
                    # Show participation percentage
                    total_time = time.time() - self.activity_tracker.discussion_start_time
                    participation_pct = (participant.total_speaking_time / max(0.1, total_time)) * 100
                    text = f"Person {person_id}: {participation_pct:.1f}% speaking"
                    color = (200, 200, 200)  # Gray for not speaking
                
                cv2.putText(output_frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 20
            
            # Add session statistics
            metrics = self.activity_tracker.get_discussion_metrics()
            stats_text = f"Quality: {metrics.discussion_quality_score:.2f} | Balance: {metrics.turn_taking_balance:.2f}"
            cv2.putText(output_frame, stats_text, (10, output_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        except Exception as e:
            # If activity tracking fails, just show basic info
            cv2.putText(output_frame, f"Frame: {self.frame_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return output_frame
    
    def export_session(self, filename: Optional[str] = None) -> str:
        """Export complete session data"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"integrated_session_{timestamp}.json"
        
        # Get activity data
        session_data = self.activity_tracker.export_session_data()
        
        # Add face tracking data
        session_data['face_tracking'] = {
            'total_people_seen': len(self.face_tracker.get_all_people()),
            'currently_present': len(self.face_tracker.get_present_people()),
            'max_people_limit': self.face_tracker.max_people,
            'frames_processed': self.frame_count,
            'people_details': {}
        }
        
        # Export individual person details
        for person_id, person in self.face_tracker.get_all_people().items():
            session_data['face_tracking']['people_details'][person_id] = {
                'name': person.name,
                'total_appearances': person.total_appearances,
                'currently_present': person.is_present,
                'last_seen': person.last_seen,
                'avg_confidence': float(np.mean(list(person.confidence_history))) if person.confidence_history else 0.0
            }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        print(f"✅ Session data exported to: {filename}")
        return filename
    
    def reset_session(self):
        """Reset both tracking systems"""
        self.face_tracker.reset_tracking()
        self.activity_tracker.reset_session()
        self.frame_count = 0
        print("🔄 Session reset completed")
    
    def get_summary(self) -> Dict:
        """Get current session summary"""
        try:
            metrics = self.activity_tracker.get_discussion_metrics()
            present_people = self.face_tracker.get_present_people()
            all_people = self.face_tracker.get_all_people()
            
            return {
                'discussion_duration': metrics.discussion_duration,
                'people_present': len(present_people),
                'total_people_seen': len(all_people),
                'discussion_quality': metrics.discussion_quality_score,
                'most_active': getattr(metrics, 'most_active_participant', 'Unknown'),
                'turn_taking_balance': metrics.turn_taking_balance,
                'frames_processed': self.frame_count
            }
        except Exception as e:
            print(f"Error getting summary: {e}")
            return {
                'discussion_duration': 0.0,
                'people_present': len(self.face_tracker.get_present_people()),
                'total_people_seen': len(self.face_tracker.get_all_people()),
                'discussion_quality': 0.0,
                'most_active': 'Unknown',
                'turn_taking_balance': 0.0,
                'frames_processed': self.frame_count
            }
    
    def get_present_people_list(self) -> List[Dict]:
        """Get list of currently present people with their activity status"""
        present = []
        present_people = self.face_tracker.get_present_people()
        participants = self.activity_tracker.participants
        
        for person_id, person in present_people.items():
            participant_info = participants.get(person_id)
            
            person_data = {
                'id': person_id,
                'name': person.name,
                'appearances': person.total_appearances,
                'confidence': float(np.mean(list(person.confidence_history))) if person.confidence_history else 0.0,
                'is_speaking': (participant_info.current_session_start is not None) if participant_info else False,
                'speaking_time': participant_info.total_speaking_time if participant_info else 0.0,
                'speaking_turns': participant_info.speaking_turns if participant_info else 0
            }
            present.append(person_data)
        
        return present