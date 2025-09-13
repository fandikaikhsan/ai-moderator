"""
Face Detection and Tracking Module
Handles face detection, tracking, and identification of participants in the discussion
Uses OpenCV's built-in face detection (compatible with all Python versions)
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass

@dataclass
class Face:
    """Represents a detected face with tracking information"""
    id: int
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    landmarks: Optional[np.ndarray]  # Will be None for OpenCV-only detection
    confidence: float
    last_seen: float
    center: Tuple[int, int]
    
class FaceTracker:
    """Face detection and tracking using OpenCV"""
    
    def __init__(self):
        # Initialize OpenCV face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Alternative: Use DNN face detection for better accuracy
        self.use_dnn = False
        try:
            # Load DNN model for more accurate detection
            self.net = cv2.dnn.readNetFromTensorflow('opencv_face_detector_uint8.pb', 'opencv_face_detector.pbtxt')
            self.use_dnn = True
            print("Using DNN face detection")
        except:
            print("Using Haar cascade face detection")
        
        # Face tracking parameters
        self.tracked_faces = {}
        self.next_face_id = 1
        self.distance_threshold = 80  # Reduced for better tracking
        self.face_timeout = 3.0  # seconds
        
        # Enhanced tracking parameters
        self.size_similarity_threshold = 0.3  # 30% size difference allowed
        self.overlap_threshold = 0.3  # 30% overlap required
        self.face_history_length = 5  # Keep history for better tracking
        self.face_histories = {}  # Store position/size history for each face
        
    def detect_faces(self, frame: np.ndarray) -> List[Face]:
        """Detect faces in the current frame"""
        height, width = frame.shape[:2]
        current_time = time.time()
        
        detected_faces = []
        
        if self.use_dnn:
            # Use DNN detection
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
            self.net.setInput(blob)
            detections = self.net.forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > 0.5:  # Confidence threshold
                    x1 = int(detections[0, 0, i, 3] * width)
                    y1 = int(detections[0, 0, i, 4] * height)
                    x2 = int(detections[0, 0, i, 5] * width)
                    y2 = int(detections[0, 0, i, 6] * height)
                    
                    x = x1
                    y = y1
                    w = x2 - x1
                    h = y2 - y1
                    
                    center = (x + w // 2, y + h // 2)
                    face_id = self._track_face(center, (x, y, w, h), current_time)
                    
                    face = Face(
                        id=face_id,
                        bbox=(x, y, w, h),
                        landmarks=self._extract_mouth_region(frame, x, y, w, h),
                        confidence=confidence,
                        last_seen=current_time,
                        center=center
                    )
                    
                    detected_faces.append(face)
                    self.tracked_faces[face_id] = face
        else:
            # Use Haar cascade detection (optimized)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Resize frame for faster detection, then scale back
            scale_factor = 0.5  # Process at half resolution for speed
            small_gray = cv2.resize(gray, (0, 0), fx=scale_factor, fy=scale_factor)
            
            # Optimized detectMultiScale parameters for speed
            faces = self.face_cascade.detectMultiScale(
                small_gray, 
                scaleFactor=1.2,  # Increased for speed
                minNeighbors=3,   # Reduced for speed
                minSize=(20, 20), # Smaller minimum for scaled image
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Scale detection results back to original size
            faces = [(int(x/scale_factor), int(y/scale_factor), 
                     int(w/scale_factor), int(h/scale_factor)) for (x, y, w, h) in faces]
            
            for (x, y, w, h) in faces:
                center = (x + w // 2, y + h // 2)
                face_id = self._track_face(center, (x, y, w, h), current_time)
                
                face = Face(
                    id=face_id,
                    bbox=(x, y, w, h),
                    landmarks=self._extract_mouth_region(frame, x, y, w, h),
                    confidence=0.8,  # Approximate confidence for Haar cascades
                    last_seen=current_time,
                    center=center
                )
                
                detected_faces.append(face)
                self.tracked_faces[face_id] = face
        
        # Remove old faces
        self._cleanup_old_faces(current_time)
        
        return detected_faces
    
    def _extract_mouth_region(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """
        Extract approximate mouth region landmarks for compatibility
        Since we don't have MediaPipe, we'll estimate mouth position
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
    
    def _track_face(self, center: Tuple[int, int], bbox: Tuple[int, int, int, int], current_time: float) -> int:
        """Enhanced face tracking using multiple criteria"""
        x, y, w, h = bbox
        
        # Find best matching existing face
        best_match_id = None
        best_match_score = float('inf')
        
        for face_id, face in self.tracked_faces.items():
            score = self._calculate_face_similarity(center, bbox, face, current_time)
            
            if score < best_match_score and score < 1.0:  # Score threshold for matching
                best_match_score = score
                best_match_id = face_id
        
        if best_match_id is not None:
            # Update the tracked face
            self._update_face_history(best_match_id, center, bbox, current_time)
            return best_match_id
        else:
            # Create new face
            face_id = self.next_face_id
            self.next_face_id += 1
            self._initialize_face_history(face_id, center, bbox, current_time)
            return face_id
    
    def _calculate_face_similarity(self, center: Tuple[int, int], bbox: Tuple[int, int, int, int], 
                                  tracked_face: Face, current_time: float) -> float:
        """Calculate similarity score between detected face and tracked face (lower = more similar)"""
        x, y, w, h = bbox
        tx, ty, tw, th = tracked_face.bbox
        
        # 1. Distance score (normalized by face size)
        distance = np.linalg.norm(np.array(center) - np.array(tracked_face.center))
        avg_size = (w + h + tw + th) / 4
        distance_score = distance / max(avg_size, 1)
        
        # 2. Size similarity score
        size_ratio = max(w/max(tw, 1), tw/max(w, 1)) + max(h/max(th, 1), th/max(h, 1))
        size_score = abs(size_ratio - 2.0)  # Perfect match would be 2.0
        
        # 3. Bounding box overlap score
        overlap_score = 1.0 - self._calculate_bbox_overlap(bbox, tracked_face.bbox)
        
        # 4. Time penalty (faces not seen recently are less likely to match)
        time_penalty = min((current_time - tracked_face.last_seen) / self.face_timeout, 1.0)
        
        # 5. Movement consistency (if we have history)
        movement_score = 0.0
        if tracked_face.id in self.face_histories and len(self.face_histories[tracked_face.id]) > 1:
            movement_score = self._calculate_movement_consistency(tracked_face.id, center)
        
        # Combine scores with weights
        total_score = (
            distance_score * 0.4 +      # 40% weight on distance
            size_score * 0.2 +          # 20% weight on size similarity
            overlap_score * 0.2 +       # 20% weight on overlap
            time_penalty * 0.1 +        # 10% weight on time
            movement_score * 0.1        # 10% weight on movement consistency
        )
        
        return total_score
    
    def _calculate_bbox_overlap(self, bbox1: Tuple[int, int, int, int], 
                               bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)
        
        if left < right and top < bottom:
            intersection = (right - left) * (bottom - top)
            union = w1 * h1 + w2 * h2 - intersection
            return intersection / max(union, 1)
        return 0.0
    
    def _calculate_movement_consistency(self, face_id: int, new_center: Tuple[int, int]) -> float:
        """Calculate how consistent the movement is with previous trajectory"""
        if face_id not in self.face_histories or len(self.face_histories[face_id]) < 2:
            return 0.0
        
        history = self.face_histories[face_id]
        
        # Calculate average velocity from recent history
        recent_positions = [h['center'] for h in history[-3:]]  # Last 3 positions
        
        if len(recent_positions) < 2:
            return 0.0
        
        # Calculate expected position based on velocity
        velocities = []
        for i in range(1, len(recent_positions)):
            vel_x = recent_positions[i][0] - recent_positions[i-1][0]
            vel_y = recent_positions[i][1] - recent_positions[i-1][1]
            velocities.append((vel_x, vel_y))
        
        if velocities:
            avg_vel_x = sum(v[0] for v in velocities) / len(velocities)
            avg_vel_y = sum(v[1] for v in velocities) / len(velocities)
            
            expected_center = (
                recent_positions[-1][0] + avg_vel_x,
                recent_positions[-1][1] + avg_vel_y
            )
            
            # Calculate how far off the prediction is
            prediction_error = np.linalg.norm(np.array(new_center) - np.array(expected_center))
            
            # Normalize by average face size
            avg_size = sum(h['size'] for h in history[-3:]) / len(history[-3:])
            movement_score = prediction_error / max(avg_size, 1)
            
            return min(movement_score / 2.0, 1.0)  # Cap at 1.0
        
        return 0.0
    
    def _initialize_face_history(self, face_id: int, center: Tuple[int, int], 
                                bbox: Tuple[int, int, int, int], current_time: float):
        """Initialize history tracking for a new face"""
        x, y, w, h = bbox
        self.face_histories[face_id] = [{
            'center': center,
            'bbox': bbox,
            'size': (w + h) / 2,
            'timestamp': current_time
        }]
    
    def _update_face_history(self, face_id: int, center: Tuple[int, int], 
                            bbox: Tuple[int, int, int, int], current_time: float):
        """Update history tracking for an existing face"""
        x, y, w, h = bbox
        
        if face_id not in self.face_histories:
            self.face_histories[face_id] = []
        
        # Add new entry
        self.face_histories[face_id].append({
            'center': center,
            'bbox': bbox,
            'size': (w + h) / 2,
            'timestamp': current_time
        })
        
        # Keep only recent history
        if len(self.face_histories[face_id]) > self.face_history_length:
            self.face_histories[face_id].pop(0)
    
    def _cleanup_old_faces(self, current_time: float):
        """Remove faces that haven't been seen recently"""
        to_remove = []
        for face_id, face in self.tracked_faces.items():
            if current_time - face.last_seen > self.face_timeout:
                to_remove.append(face_id)
        
        for face_id in to_remove:
            del self.tracked_faces[face_id]
            # Also clean up history
            if face_id in self.face_histories:
                del self.face_histories[face_id]
    
    def draw_faces(self, frame: np.ndarray, faces: List[Face]) -> np.ndarray:
        """Draw face bounding boxes and IDs on the frame with enhanced tracking info"""
        for face in faces:
            x, y, w, h = face.bbox
            
            # Choose color based on face ID for consistency
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                     (255, 0, 255), (0, 255, 255), (128, 255, 128), (255, 128, 128)]
            color = colors[face.id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw face ID with background for better visibility
            label = f"Person {face.id}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background rectangle for text
            cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0] + 10, y), color, -1)
            
            # Text
            cv2.putText(frame, label, (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw tracking history if available
            if face.id in self.face_histories and len(self.face_histories[face.id]) > 1:
                history = self.face_histories[face.id]
                # Draw recent movement trail
                for i in range(1, min(len(history), 5)):
                    pt1 = history[i-1]['center']
                    pt2 = history[i]['center']
                    cv2.line(frame, pt1, pt2, color, 1)
            
            # Draw confidence
            cv2.putText(frame, f"{face.confidence:.2f}", 
                       (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (255, 255, 255), 1)
            
            # Draw center point
            cv2.circle(frame, face.center, 3, color, -1)
        
        # Add tracking info overlay
        if faces:
            info_text = f"Tracking {len(faces)} person(s)"
            cv2.putText(frame, info_text, (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def get_active_faces(self) -> List[Face]:
        """Get currently tracked faces"""
        return list(self.tracked_faces.values())
    
    def get_face_count(self) -> int:
        """Get number of currently tracked faces"""
        return len(self.tracked_faces)