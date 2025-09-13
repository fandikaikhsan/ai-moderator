"""
Enhanced Face Detection and Person Tracking with DNN and Haar Cascade Methods
Tracks individuals entering/leaving frame with persistent identity
Supports DNN and Haar Cascade detection methods for optimal performance
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import math

@dataclass
class Person:
    """Represents a tracked person"""
    person_id: int
    name: str
    face_encoding: Optional[np.ndarray] = None  # Face features for recognition
    last_seen: float = 0.0
    total_appearances: int = 0
    current_bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    is_present: bool = False
    confidence_history: Optional[deque] = None
    center_history: Optional[deque] = None  # Track face movement
    
    def __post_init__(self):
        if self.confidence_history is None:
            self.confidence_history = deque(maxlen=10)
        if self.center_history is None:
            self.center_history = deque(maxlen=5)

@dataclass
class FaceDetection:
    """Single face detection result"""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    face_roi: np.ndarray  # Cropped face region
    center: Tuple[int, int]
    features: Optional[np.ndarray] = None  # Face features for matching

class EnhancedFaceTracker:
    """Enhanced face tracker with person identity management and multiple detection methods"""
    
    def __init__(self, max_people: int = 8, recognition_threshold: float = 0.5, detection_method: str = "dnn"):
        """
        Initialize enhanced face tracker with multiple detection methods
        
        Args:
            max_people: Maximum number of people to track
            recognition_threshold: Similarity threshold for person recognition (0-1)
            detection_method: Detection method ("dnn", "haar")
        """
        self.max_people = max_people
        self.recognition_threshold = recognition_threshold
        self.detection_method = detection_method.lower()
        
        # Initialize detection methods
        self._initialize_detection_methods()
        
        # Person tracking with stable parameters
        self.people: Dict[int, Person] = {}
        self.next_person_id = 1
        self.absent_timeout = 2.0      # Shorter timeout to reduce blinking
        self.distance_threshold = 60   # Smaller distance for more stable tracking
        
        # Detection parameters optimized for stability
        self.min_face_size = (40, 40)  # Slightly larger minimum size
        self.scale_factor = 1.05       # Smaller scale factor for more detections
        self.min_neighbors = 3         # Lower neighbors for more sensitivity
        
        # Smoothing parameters
        self._previous_detections = []
        self._frame_count = 0
        
        # Speech detection tracking
        self.speech_data = {}  # Track speech information per person
    
    def _initialize_detection_methods(self):
        """Initialize all available detection methods"""
        # Initialize DNN detection
        self.net = None
        self.use_dnn = False
        try:
            self.net = cv2.dnn.readNetFromTensorflow('opencv_face_detector_uint8.pb', 'opencv_face_detector.pbtxt')
            self.use_dnn = True
            print("✅ DNN face detection available")
        except:
            print("⚠️  DNN models not found")
        
        # Initialize Haar cascade
        self.face_cascade = None
        self.use_haar = False
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if not self.face_cascade.empty():
                self.use_haar = True
                print("✅ Haar cascade face detection available")
        except:
            print("⚠️  Haar cascade not available")
        
        # Set active detection method
        self._set_active_method(self.detection_method)
    
    def _set_active_method(self, method: str):
        """Set the active detection method"""
        method = method.lower()
        
        if method == "dnn" and self.use_dnn:
            self.active_method = "dnn"
            print(f"🎯 Using DNN face detection")
        elif method == "haar" and self.use_haar:
            self.active_method = "haar"
            print(f"🎯 Using Haar cascade face detection")
        else:
            # Auto-fallback (prefer DNN for better accuracy)
            if self.use_dnn:
                self.active_method = "dnn"
                print(f"🎯 Auto-selected DNN face detection")
            elif self.use_haar:
                self.active_method = "haar"
                print(f"🎯 Auto-selected Haar cascade face detection")
            else:
                self.active_method = None
                print("❌ No face detection methods available")
    
    def set_detection_method(self, method: str):
        """Change the active detection method"""
        self._set_active_method(method)
    
    def get_available_methods(self) -> List[str]:
        """Get list of available detection methods"""
        methods = []
        if self.use_haar:
            methods.append("haar")
        if self.use_dnn:
            methods.append("dnn")
        if self.use_yolo:
            methods.append("yolo")
        return methods
        
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in frame using the active detection method
        
        Args:
            frame: Input video frame
            
        Returns:
            List of face detections with speech information (if available)
        """
        if self.active_method == "dnn" and self.use_dnn:
            return self._detect_faces_dnn(frame)
        elif self.active_method == "haar" and self.use_haar:
            return self._detect_faces_haar(frame)
        else:
            print("❌ No active detection method available")
            return []
    
    def _detect_faces_dnn(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces using OpenCV DNN"""
        height, width = frame.shape[:2]
        detections = []
        
        try:
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
            self.net.setInput(blob)
            dnn_detections = self.net.forward()
            
            for i in range(dnn_detections.shape[2]):
                confidence = dnn_detections[0, 0, i, 2]
                
                if confidence > 0.5:  # Confidence threshold
                    x1 = int(dnn_detections[0, 0, i, 3] * width)
                    y1 = int(dnn_detections[0, 0, i, 4] * height)
                    x2 = int(dnn_detections[0, 0, i, 5] * width)
                    y2 = int(dnn_detections[0, 0, i, 6] * height)
                    
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1
                    
                    # Ensure bbox is within frame bounds
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, width - x)
                    h = min(h, height - y)
                    
                    if w > 20 and h > 20:  # Minimum face size
                        center = (x + w // 2, y + h // 2)
                        face_roi = frame[y:y+h, x:x+w] if y+h <= height and x+w <= width else frame[y:height, x:width]
                        
                        # Extract features
                        features = self._extract_simple_features(face_roi)
                        
                        # Detect speech
                        is_speaking, speaking_intensity = self._detect_speech_simple(frame, (x, y, w, h), i)
                        
                        detection = FaceDetection(
                            bbox=(x, y, w, h),
                            confidence=confidence,
                            face_roi=face_roi,
                            center=center,
                            features=features
                        )
                        
                        # Add speech information
                        detection.is_speaking = is_speaking
                        detection.speaking_intensity = speaking_intensity
                        
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"DNN detection error: {e}")
            return []
    
    def _detect_faces_haar(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces using Haar cascade"""
        height, width = frame.shape[:2]
        detections = []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_face_size
            )
            
            for i, (x, y, w, h) in enumerate(faces):
                # Calculate confidence based on face size
                confidence = min(1.0, (w * h) / (100 * 100))
                center = (x + w // 2, y + h // 2)
                
                # Extract face ROI
                face_roi = frame[y:y+h, x:x+w] if y+h <= height and x+w <= width else frame[y:height, x:width]
                
                # Extract features
                features = self._extract_simple_features(face_roi)
                
                # Detect speech
                is_speaking, speaking_intensity = self._detect_speech_simple(frame, (x, y, w, h), i)
                
                detection = FaceDetection(
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    face_roi=face_roi,
                    center=center,
                    features=features
                )
                
                # Add speech information
                detection.is_speaking = is_speaking
                detection.speaking_intensity = speaking_intensity
                
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"Haar cascade detection error: {e}")
            return []

    def _detect_speech_simple(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], face_id: int) -> Tuple[bool, float]:
        """
        Simple speech detection using mouth region analysis (like the working 51.7% method)
        
        Args:
            frame: Input frame
            bbox: Face bounding box (x, y, w, h)
            face_id: Face identifier for tracking
            
        Returns:
            Tuple of (is_speaking, intensity)
        """
        try:
            x, y, w, h = bbox
            
            # Define mouth region (lower third of face)
            mouth_y = y + int(h * 0.65)
            mouth_x = x + int(w * 0.25)
            mouth_w = int(w * 0.5)
            mouth_h = int(h * 0.25)
            
            # Ensure mouth region is within frame bounds
            mouth_y = max(0, min(mouth_y, frame.shape[0] - mouth_h))
            mouth_x = max(0, min(mouth_x, frame.shape[1] - mouth_w))
            mouth_h = min(mouth_h, frame.shape[0] - mouth_y)
            mouth_w = min(mouth_w, frame.shape[1] - mouth_x)
            
            if mouth_h <= 0 or mouth_w <= 0:
                return False, 0.0
            
            # Extract mouth region
            mouth_roi = frame[mouth_y:mouth_y+mouth_h, mouth_x:mouth_x+mouth_w]
            
            if mouth_roi.size == 0:
                return False, 0.0
            
            # Convert to grayscale for analysis
            mouth_gray = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
            
            # Simple activity detection based on pixel variance (like working method)
            variance = np.var(mouth_gray)
            
            # Track variance changes (simple temporal analysis)
            face_key = f"face_{face_id}"
            if not hasattr(self, 'mouth_variances'):
                self.mouth_variances = {}
            
            if face_key not in self.mouth_variances:
                self.mouth_variances[face_key] = deque(maxlen=5)
            
            self.mouth_variances[face_key].append(variance)
            
            # Calculate activity intensity
            if len(self.mouth_variances[face_key]) >= 2:
                recent_variances = list(self.mouth_variances[face_key])
                variance_change = max(recent_variances) - min(recent_variances)
                
                # Normalize intensity (empirically determined thresholds)
                intensity = min(1.0, variance_change / 500.0)
                
                # Simple speaking detection (similar to 51.7% success rate method)
                is_speaking = variance > 800 and intensity > 0.1
                
                return is_speaking, intensity
            
            return False, 0.0
            
        except Exception as e:
            print(f"Speech detection error: {e}")
            return False, 0.0
    
    def _extract_simple_features(self, face_roi: np.ndarray) -> np.ndarray:
        """
        Extract simple features from face ROI for recognition
        
        Args:
            face_roi: Cropped face region
            
        Returns:
            Feature vector
        """
        if face_roi.size == 0:
            return np.array([])
        
        try:
            # Resize to standard size
            face_resized = cv2.resize(face_roi, (64, 64))
            
            # Convert to grayscale
            if len(face_resized.shape) == 3:
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_resized
            
            # Simple histogram-based features (more robust than HOG for this use case)
            hist = cv2.calcHist([face_gray], [0], None, [32], [0, 256])
            
            # Add some spatial features
            # Divide face into quadrants and compute mean intensity
            h, w = face_gray.shape
            quadrants = [
                face_gray[0:h//2, 0:w//2].mean(),        # Top-left
                face_gray[0:h//2, w//2:w].mean(),        # Top-right
                face_gray[h//2:h, 0:w//2].mean(),        # Bottom-left
                face_gray[h//2:h, w//2:w].mean(),        # Bottom-right
            ]
            
            # Combine histogram and spatial features
            features = np.concatenate([hist.flatten(), np.array(quadrants)])
            
            return features
        
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.array([])
    
    def _calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Calculate similarity between two feature vectors
        
        Args:
            features1, features2: Feature vectors to compare
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        if features1.size == 0 or features2.size == 0:
            return 0.0
        
        if len(features1) != len(features2):
            return 0.0
        
        try:
            # Normalize features
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            features1_norm = features1 / norm1
            features2_norm = features2 / norm2
            
            # Calculate cosine similarity
            similarity = np.dot(features1_norm, features2_norm)
            
            # Convert to 0-1 range
            return max(0.0, (similarity + 1) / 2)
        
        except Exception as e:
            print(f"Similarity calculation error: {e}")
            return 0.0
    
    def _calculate_distance(self, center1: Tuple[int, int], center2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _smooth_detections(self, detections: List[FaceDetection]) -> List[FaceDetection]:
        """
        Apply temporal smoothing to detections to reduce jitter
        
        Args:
            detections: Current frame detections
            
        Returns:
            Smoothed detections
        """
        if not hasattr(self, '_previous_detections'):
            self._previous_detections = []
        
        smoothed_detections = []
        
        for detection in detections:
            # Find closest previous detection
            closest_prev = None
            min_distance = float('inf')
            
            for prev_det in self._previous_detections:
                distance = self._calculate_distance(detection.center, prev_det.center)
                if distance < min_distance and distance < 50:  # 50 pixel threshold
                    min_distance = distance
                    closest_prev = prev_det
            
            # Apply smoothing if we found a close previous detection
            if closest_prev is not None:
                # Smooth the bounding box
                smooth_factor = 0.7
                x1, y1, w1, h1 = detection.bbox
                x2, y2, w2, h2 = closest_prev.bbox
                
                smoothed_x = int(smooth_factor * x2 + (1 - smooth_factor) * x1)
                smoothed_y = int(smooth_factor * y2 + (1 - smooth_factor) * y1)
                smoothed_w = int(smooth_factor * w2 + (1 - smooth_factor) * w1)
                smoothed_h = int(smooth_factor * h2 + (1 - smooth_factor) * h1)
                
                detection.bbox = (smoothed_x, smoothed_y, smoothed_w, smoothed_h)
                detection.center = (smoothed_x + smoothed_w // 2, smoothed_y + smoothed_h // 2)
            
            smoothed_detections.append(detection)
        
        # Store current detections for next frame
        self._previous_detections = smoothed_detections.copy()
        
        return smoothed_detections
    
    def update_tracking(self, frame: np.ndarray) -> Dict[int, Person]:
        """
        Update person tracking with new frame
        
        Args:
            frame: Current video frame
            
        Returns:
            Dictionary of currently tracked people
        """
        current_time = time.time()
        
        # Detect faces in current frame
        detections = self.detect_faces(frame)
        
        # Track which people are matched this frame
        matched_people = set()
        unmatched_detections = detections.copy()
        
        # Try to match detections to existing people
        for detection in detections[:]:  # Use slice to avoid modification during iteration
            best_match_id = None
            best_score = 0.0
            
            # Compare with all known people
            for person_id, person in self.people.items():
                if not person.is_present:
                    continue  # Skip absent people for initial matching
                
                # Calculate feature similarity
                feature_similarity = 0.0
                if person.face_encoding is not None and detection.features.size > 0:
                    feature_similarity = self._calculate_similarity(detection.features, person.face_encoding)
                
                # Calculate distance similarity (if person has recent position)
                distance_similarity = 0.0
                if person.center_history and len(person.center_history) > 0:
                    last_center = person.center_history[-1]
                    distance = self._calculate_distance(detection.center, last_center)
                    distance_similarity = max(0.0, 1.0 - (distance / self.distance_threshold))
                
                # Combine similarities (weight features more heavily)
                combined_score = (feature_similarity * 0.7 + distance_similarity * 0.3)
                
                if combined_score > best_score and combined_score > self.recognition_threshold:
                    best_score = combined_score
                    best_match_id = person_id
            
            # If no good match with present people, try absent people with higher threshold
            if best_match_id is None:
                for person_id, person in self.people.items():
                    if person.is_present:
                        continue  # Already checked present people
                    
                    if person.face_encoding is not None and detection.features.size > 0:
                        feature_similarity = self._calculate_similarity(detection.features, person.face_encoding)
                        
                        # Higher threshold for returning people
                        if feature_similarity > best_score and feature_similarity > (self.recognition_threshold + 0.1):
                            best_score = feature_similarity
                            best_match_id = person_id
            
            # Update matched person or create new one
            if best_match_id is not None:
                # Update existing person
                person = self.people[best_match_id]
                person.last_seen = current_time
                person.current_bbox = detection.bbox
                person.is_present = True
                person.total_appearances += 1
                person.confidence_history.append(detection.confidence)
                person.center_history.append(detection.center)
                
                # Update face encoding (running average)
                if detection.features.size > 0:
                    if person.face_encoding is not None:
                        # Weighted average: 80% old, 20% new (more conservative)
                        person.face_encoding = 0.8 * person.face_encoding + 0.2 * detection.features
                    else:
                        person.face_encoding = detection.features.copy()
                
                # Update speech information if available (YOLOv11)
                if hasattr(detection, 'is_speaking') and hasattr(detection, 'speaking_intensity'):
                    self.speech_data[best_match_id] = {
                        'is_speaking': detection.is_speaking,
                        'speaking_intensity': detection.speaking_intensity,
                        'last_update': current_time,
                        'mouth_landmarks': getattr(detection, 'mouth_landmarks', None)
                    }
                
                matched_people.add(best_match_id)
                if detection in unmatched_detections:
                    unmatched_detections.remove(detection)
            
            elif len(self.people) < self.max_people:
                # Create new person
                person_id = self.next_person_id
                self.next_person_id += 1
                
                new_person = Person(
                    person_id=person_id,
                    name=f"Person {person_id}",
                    face_encoding=detection.features.copy() if detection.features.size > 0 else None,
                    last_seen=current_time,
                    total_appearances=1,
                    current_bbox=detection.bbox,
                    is_present=True
                )
                new_person.confidence_history.append(detection.confidence)
                new_person.center_history.append(detection.center)
                
                self.people[person_id] = new_person
                
                # Initialize speech information if available (YOLOv11)
                if hasattr(detection, 'is_speaking') and hasattr(detection, 'speaking_intensity'):
                    self.speech_data[person_id] = {
                        'is_speaking': detection.is_speaking,
                        'speaking_intensity': detection.speaking_intensity,
                        'last_update': current_time,
                        'mouth_landmarks': getattr(detection, 'mouth_landmarks', None)
                    }
                
                matched_people.add(person_id)
                if detection in unmatched_detections:
                    unmatched_detections.remove(detection)
        
        # Mark unmatched people as absent (if timeout exceeded)
        for person_id, person in self.people.items():
            if person_id not in matched_people:
                time_since_seen = current_time - person.last_seen
                if time_since_seen > self.absent_timeout:
                    person.is_present = False
                    person.current_bbox = None
        
        # Return currently present people
        return {pid: person for pid, person in self.people.items() if person.is_present}
    
    def get_person_info(self, person_id: int) -> Optional[Person]:
        """Get information about a specific person"""
        return self.people.get(person_id)
    
    def get_all_people(self) -> Dict[int, Person]:
        """Get all tracked people (present and absent)"""
        return self.people.copy()
    
    def get_present_people(self) -> Dict[int, Person]:
        """Get only currently present people"""
        return {pid: person for pid, person in self.people.items() if person.is_present}
    
    def reset_tracking(self):
        """Reset all tracking data"""
        self.people.clear()
        self.next_person_id = 1
    
    def draw_tracking_info(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw tracking information on frame
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with tracking info overlaid
        """
        output_frame = frame.copy()
        
        for person_id, person in self.people.items():
            if person.is_present and person.current_bbox is not None:
                x, y, w, h = person.current_bbox
                
                # Draw bounding box
                color = (0, 255, 0)  # Green for present people
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw person info
                label = f"{person.name} (#{person.total_appearances})"
                cv2.putText(output_frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw confidence
                if person.confidence_history:
                    avg_confidence = np.mean(list(person.confidence_history))
                    conf_text = f"Conf: {avg_confidence:.2f}"
                    cv2.putText(output_frame, conf_text, (x, y + h + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Draw center point
                if person.center_history:
                    center = person.center_history[-1]
                    cv2.circle(output_frame, center, 3, (0, 0, 255), -1)
        
        # Draw summary info
        present_count = len(self.get_present_people())
        total_count = len(self.people)
        summary = f"Present: {present_count}/{total_count} (Max: {self.max_people})"
        cv2.putText(output_frame, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return output_frame
    
    def get_speech_info(self, person_id: int) -> Dict:
        """Get speech information for a specific person"""
        return self.speech_data.get(person_id, {
            'is_speaking': False,
            'speaking_intensity': 0.0,
            'last_update': 0.0,
            'mouth_landmarks': None
        })
    
    def get_all_speech_info(self) -> Dict[int, Dict]:
        """Get speech information for all tracked people"""
        return self.speech_data.copy()
    
    def is_person_speaking(self, person_id: int) -> bool:
        """Check if a specific person is currently speaking"""
        speech_info = self.get_speech_info(person_id)
        current_time = time.time()
        
        # Consider speech info stale after 1 second
        if current_time - speech_info.get('last_update', 0) > 1.0:
            return False
        
        return speech_info.get('is_speaking', False)
    
    def get_speaking_people(self) -> List[int]:
        """Get list of person IDs who are currently speaking"""
        speaking = []
        current_time = time.time()
        
        for person_id, speech_info in self.speech_data.items():
            if (current_time - speech_info.get('last_update', 0) <= 1.0 and 
                speech_info.get('is_speaking', False)):
                speaking.append(person_id)
        
        return speaking
    
    def cleanup_old_speech_data(self, max_age: float = 10.0):
        """Clean up old speech tracking data"""
        current_time = time.time()
        to_remove = []
        
        for person_id, speech_info in self.speech_data.items():
            if current_time - speech_info.get('last_update', 0) > max_age:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.speech_data[person_id]
    
    def draw_tracking_info_with_speech(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw tracking information on frame including speech indicators
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with tracking and speech info overlaid
        """
        output_frame = frame.copy()
        
        for person_id, person in self.people.items():
            if person.is_present and person.current_bbox is not None:
                x, y, w, h = person.current_bbox
                
                # Get speech info
                speech_info = self.get_speech_info(person_id)
                is_speaking = self.is_person_speaking(person_id)
                
                # Draw bounding box with speech indication
                if is_speaking:
                    color = (0, 255, 255)  # Yellow for speaking
                    thickness = 3
                else:
                    color = (0, 255, 0)  # Green for present
                    thickness = 2
                
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), color, thickness)
                
                # Draw person info
                label = f"{person.name}"
                if is_speaking:
                    intensity = speech_info.get('speaking_intensity', 0.0)
                    label += f" 🎤 ({intensity:.2f})"
                
                cv2.putText(output_frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw mouth landmarks if available (YOLOv11)
                mouth_landmarks = speech_info.get('mouth_landmarks')
                if mouth_landmarks is not None and is_speaking:
                    for point in mouth_landmarks:
                        cv2.circle(output_frame, tuple(map(int, point)), 2, (255, 0, 255), -1)
                
                # Draw confidence
                if person.confidence_history:
                    avg_confidence = np.mean(list(person.confidence_history))
                    conf_text = f"Conf: {avg_confidence:.2f}"
                    cv2.putText(output_frame, conf_text, (x, y + h + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw summary info with speech count
        present_count = len(self.get_present_people())
        speaking_count = len(self.get_speaking_people())
        total_count = len(self.people)
        
        summary = f"Present: {present_count}/{total_count} | Speaking: {speaking_count}"
        cv2.putText(output_frame, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show detection method
        method_text = f"Method: {self.active_method.upper() if self.active_method else 'None'}"
        cv2.putText(output_frame, method_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return output_frame
    
    def _extract_mouth_region(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """
        Extract approximate mouth region landmarks for compatibility with existing system
        """
        # Estimate mouth position (lower third of face, centered)
        mouth_y = y + int(h * 0.65)  # Mouth is roughly 65% down the face
        mouth_x_center = x + w // 2
        mouth_width = int(w * 0.4)   # Mouth is roughly 40% of face width
        mouth_height = int(h * 0.15) # Mouth is roughly 15% of face height
        
        # Create simple mouth landmarks (rectangle approximation)
        mouth_landmarks = np.array([
            [mouth_x_center - mouth_width//2, mouth_y - mouth_height//2],  # Top left
            [mouth_x_center + mouth_width//2, mouth_y - mouth_height//2],  # Top right
            [mouth_x_center + mouth_width//2, mouth_y + mouth_height//2],  # Bottom right
            [mouth_x_center - mouth_width//2, mouth_y + mouth_height//2],  # Bottom left
            [mouth_x_center, mouth_y - mouth_height//2],                   # Top center
            [mouth_x_center, mouth_y + mouth_height//2],                   # Bottom center
            [mouth_x_center - mouth_width//2, mouth_y],                    # Left center
            [mouth_x_center + mouth_width//2, mouth_y],                    # Right center
        ])
        
        return mouth_landmarks