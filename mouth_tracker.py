"""
Mouth Movement Detection Module
Analyzes facial landmarks to detect when people are speaking
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from dataclasses import dataclass
from collections import deque
from face_tracker import Face

@dataclass
class SpeechActivity:
    """Represents speech activity for a person"""
    person_id: int
    is_speaking: bool
    speaking_intensity: float  # 0-1 scale
    mouth_aspect_ratio: float
    lip_distance: float
    speaking_duration: float
    last_speech_time: float

class MouthTracker:
    """Tracks mouth movements to detect speech activity"""
    
    # Simplified landmark indices for OpenCV-based detection
    MOUTH_LANDMARKS = [0, 1, 2, 3, 4, 5, 6, 7]  # All 8 mouth points
    
    # Upper and lower lip landmarks for MAR calculation
    UPPER_LIP = [4, 7]  # Top center, right center
    LOWER_LIP = [5, 6]  # Bottom center, left center
    
    # Left and right corner landmarks for lip distance
    LIP_CORNERS = [6, 7]  # left center, right center
    
    def __init__(self, history_length: int = 15):
        """
        Initialize mouth tracker
        
        Args:
            history_length: Number of frames to keep in movement history
        """
        self.history_length = history_length
        self.mouth_histories: Dict[int, deque] = {}
        self.speech_states: Dict[int, SpeechActivity] = {}
        
        # Thresholds for speech detection (adjusted for simpler landmarks)
        self.mar_threshold = 0.02  # Lower threshold for simplified detection
        self.movement_threshold = 0.015  # Movement threshold
        self.min_speaking_frames = 3  # Minimum frames to consider as speaking
        self.speaking_timeout = 0.5  # seconds
        
    def analyze_mouth_movement(self, faces: List[Face]) -> Dict[int, SpeechActivity]:
        """
        Analyze mouth movement for all detected faces
        
        Args:
            faces: List of detected faces with landmarks
            
        Returns:
            Dictionary mapping person ID to speech activity
        """
        current_time = time.time()
        current_activities = {}
        
        for face in faces:
            # Check if landmarks exist and are valid
            try:
                if face.landmarks is not None and hasattr(face.landmarks, '__len__') and len(face.landmarks) > 0:
                    activity = self._analyze_single_mouth(face, current_time)
                    current_activities[face.id] = activity
                    self.speech_states[face.id] = activity
                else:
                    # Create default activity for faces without landmarks
                    current_activities[face.id] = SpeechActivity(
                        person_id=face.id,
                        is_speaking=False,
                        speaking_intensity=0.0,
                        mouth_aspect_ratio=0.0,
                        lip_distance=0.0,
                        speaking_duration=0.0,
                        last_speech_time=current_time
                    )
            except Exception as e:
                # Handle any landmark processing errors gracefully
                current_activities[face.id] = SpeechActivity(
                    person_id=face.id,
                    is_speaking=False,
                    speaking_intensity=0.0,
                    mouth_aspect_ratio=0.0,
                    lip_distance=0.0,
                    speaking_duration=0.0,
                    last_speech_time=current_time
                )
        
        # Update speaking durations
        self._update_speaking_durations(current_time)
        
        return current_activities
    
    def _analyze_single_mouth(self, face: Face, current_time: float) -> SpeechActivity:
        """Analyze mouth movement for a single face"""
        person_id = face.id
        landmarks = face.landmarks
        
        # Initialize history if needed
        if person_id not in self.mouth_histories:
            self.mouth_histories[person_id] = deque(maxlen=self.history_length)
        
        # Calculate mouth features (simplified for OpenCV landmarks)
        mar = self._calculate_mouth_aspect_ratio_simple(landmarks)
        lip_distance = self._calculate_lip_distance_simple(landmarks)
        movement_intensity = self._calculate_movement_intensity(person_id, landmarks)
        
        # Store current mouth state
        mouth_state = {
            'mar': mar,
            'lip_distance': lip_distance,
            'landmarks': landmarks.copy() if landmarks is not None else None,
            'timestamp': current_time
        }
        self.mouth_histories[person_id].append(mouth_state)
        
        # Determine if speaking
        is_speaking = self._is_speaking(person_id, mar, movement_intensity)
        speaking_intensity = self._calculate_speaking_intensity(mar, movement_intensity)
        
        # Get previous state or create new one
        prev_state = self.speech_states.get(person_id)
        speaking_duration = 0
        last_speech_time = current_time if is_speaking else (
            prev_state.last_speech_time if prev_state else 0
        )
        
        if prev_state:
            if is_speaking:
                if prev_state.is_speaking:
                    speaking_duration = prev_state.speaking_duration + (current_time - prev_state.last_speech_time)
                else:
                    speaking_duration = 0.1  # Start counting
            else:
                speaking_duration = prev_state.speaking_duration
        
        return SpeechActivity(
            person_id=person_id,
            is_speaking=is_speaking,
            speaking_intensity=speaking_intensity,
            mouth_aspect_ratio=mar,
            lip_distance=lip_distance,
            speaking_duration=speaking_duration,
            last_speech_time=last_speech_time
        )
    
    def _calculate_mouth_aspect_ratio_simple(self, landmarks: np.ndarray) -> float:
        """Calculate simplified Mouth Aspect Ratio using basic landmarks"""
        try:
            # Use simplified mouth landmarks (rectangle approximation)
            if len(landmarks) < 8:
                return 0.0
            
            # Calculate vertical distances (top to bottom)
            top_center = landmarks[4]    # Top center
            bottom_center = landmarks[5] # Bottom center
            vertical_dist = np.linalg.norm(top_center - bottom_center)
            
            # Calculate horizontal distance (left to right)
            left_center = landmarks[6]   # Left center  
            right_center = landmarks[7]  # Right center
            horizontal_dist = np.linalg.norm(right_center - left_center)
            
            # Calculate MAR
            if horizontal_dist > 0:
                mar = vertical_dist / horizontal_dist
            else:
                mar = 0.0
            
            return mar
        except (IndexError, ZeroDivisionError):
            return 0.0
    
    def _calculate_lip_distance_simple(self, landmarks: np.ndarray) -> float:
        """Calculate distance between lip corners using simplified landmarks"""
        try:
            if len(landmarks) < 8:
                return 0.0
            
            left_corner = landmarks[6]   # Left center (approximating corner)
            right_corner = landmarks[7]  # Right center (approximating corner)
            distance = np.linalg.norm(right_corner - left_corner)
            return distance
        except IndexError:
            return 0.0
    
    def _calculate_movement_intensity(self, person_id: int, current_landmarks: np.ndarray) -> float:
        """Calculate mouth movement intensity based on landmark changes"""
        if person_id not in self.mouth_histories or len(self.mouth_histories[person_id]) < 2:
            return 0.0
        
        try:
            # Get previous mouth landmarks
            prev_state = list(self.mouth_histories[person_id])[-2]
            prev_landmarks = prev_state['landmarks']
            
            # Calculate movement of mouth landmarks
            mouth_current = current_landmarks[self.MOUTH_LANDMARKS]
            mouth_prev = prev_landmarks[self.MOUTH_LANDMARKS]
            
            # Calculate average movement
            movements = np.linalg.norm(mouth_current - mouth_prev, axis=1)
            avg_movement = np.mean(movements)
            
            return min(avg_movement / 10.0, 1.0)  # Normalize to 0-1
        except (IndexError, ValueError):
            return 0.0
    
    def _is_speaking(self, person_id: int, mar: float, movement_intensity: float) -> bool:
        """Determine if person is speaking based on mouth features"""
        # Check if mouth is open enough and there's movement
        mouth_open = mar > self.mar_threshold
        has_movement = movement_intensity > self.movement_threshold
        
        # Need both conditions
        if not (mouth_open and has_movement):
            return False
        
        # Check consistency across frames
        if person_id not in self.mouth_histories:
            return False
        
        history = list(self.mouth_histories[person_id])
        if len(history) < self.min_speaking_frames:
            return False
        
        # Check recent frames for consistent speaking indicators
        recent_frames = history[-self.min_speaking_frames:]
        speaking_frames = 0
        
        for frame in recent_frames:
            frame_mar = frame['mar']
            if len(history) >= 2:
                # Calculate movement for this frame
                prev_landmarks = history[history.index(frame) - 1]['landmarks'] if history.index(frame) > 0 else frame['landmarks']
                current_landmarks = frame['landmarks']
                
                try:
                    mouth_current = current_landmarks[self.MOUTH_LANDMARKS]
                    mouth_prev = prev_landmarks[self.MOUTH_LANDMARKS]
                    movements = np.linalg.norm(mouth_current - mouth_prev, axis=1)
                    frame_movement = min(np.mean(movements) / 10.0, 1.0)
                except:
                    frame_movement = 0
                
                if frame_mar > self.mar_threshold and frame_movement > self.movement_threshold:
                    speaking_frames += 1
        
        return speaking_frames >= (self.min_speaking_frames // 2)
    
    def _calculate_speaking_intensity(self, mar: float, movement_intensity: float) -> float:
        """Calculate speaking intensity (0-1 scale)"""
        # Combine MAR and movement intensity
        mar_score = min(mar / (self.mar_threshold * 3), 1.0)  # Normalize MAR
        movement_score = movement_intensity
        
        # Weighted combination
        intensity = (mar_score * 0.4 + movement_score * 0.6)
        return min(intensity, 1.0)
    
    def _update_speaking_durations(self, current_time: float):
        """Update speaking durations for all tracked people"""
        for person_id, activity in self.speech_states.items():
            if not activity.is_speaking and current_time - activity.last_speech_time > self.speaking_timeout:
                # Reset if not speaking for too long
                activity.speaking_duration = 0
    
    def draw_mouth_analysis(self, frame: np.ndarray, faces: List[Face], 
                           activities: Dict[int, SpeechActivity]) -> np.ndarray:
        """Draw mouth analysis visualization on frame"""
        for face in faces:
            if face.id in activities:
                activity = activities[face.id]
                x, y, w, h = face.bbox
                
                # Color based on speaking state
                color = (0, 255, 0) if activity.is_speaking else (128, 128, 128)
                
                # Draw speaking indicator
                status_text = "SPEAKING" if activity.is_speaking else "QUIET"
                cv2.putText(frame, status_text, (x, y - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw intensity bar
                intensity_width = int(activity.speaking_intensity * 100)
                cv2.rectangle(frame, (x, y + h + 5), 
                             (x + intensity_width, y + h + 15), color, -1)
                cv2.rectangle(frame, (x, y + h + 5), 
                             (x + 100, y + h + 15), (255, 255, 255), 1)
                
                # Draw mouth landmarks if available
                if face.landmarks is not None:
                    self._draw_mouth_landmarks(frame, face.landmarks)
        
        return frame
    
    def _draw_mouth_landmarks(self, frame: np.ndarray, landmarks: np.ndarray):
        """Draw simplified mouth landmarks on frame"""
        try:
            if len(landmarks) >= 8:
                # Draw the 8 mouth points
                for i, point in enumerate(landmarks):
                    color = (0, 255, 255) if i < 4 else (255, 255, 0)  # Different colors for corners vs centers
                    cv2.circle(frame, (int(point[0]), int(point[1])), 2, color, -1)
                
                # Draw mouth outline (connect the rectangle points)
                pts = landmarks[:4].astype(np.int32)  # First 4 points form rectangle
                cv2.polylines(frame, [pts], True, (0, 255, 255), 1)
        except (IndexError, ValueError):
            pass
    
    def get_speaking_summary(self) -> Dict[int, Dict]:
        """Get summary of speaking activity for all people"""
        summary = {}
        for person_id, activity in self.speech_states.items():
            summary[person_id] = {
                'is_speaking': activity.is_speaking,
                'speaking_duration': activity.speaking_duration,
                'speaking_intensity': activity.speaking_intensity,
                'last_speech_time': activity.last_speech_time
            }
        return summary