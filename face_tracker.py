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
        try:
            # Load DNN model for face detection (more accurate than Haar cascades)
            self.net = cv2.dnn.readNetFromTensorflow('opencv_face_detector_uint8.pb', 'opencv_face_detector.pbtxt')
            self.use_dnn = True
            print("Using DNN face detection")
        except:
            # Fallback to Haar cascades if DNN model not available
            self.use_dnn = False
            print("Using Haar cascade face detection")
        
        # Tracking variables
        self.tracked_faces: Dict[int, Face] = {}
        self.next_face_id = 0
        self.face_timeout = 2.0  # seconds
        self.distance_threshold = 100  # pixels
        
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
                    face_id = self._track_face(center, current_time)
                    
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
                face_id = self._track_face(center, current_time)
                
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
    
    def _track_face(self, center: Tuple[int, int], current_time: float) -> int:
        """Track a face or assign new ID"""
        # Find closest existing face
        min_distance = float('inf')
        closest_id = None
        
        for face_id, face in self.tracked_faces.items():
            distance = np.linalg.norm(np.array(center) - np.array(face.center))
            if distance < min_distance and distance < self.distance_threshold:
                min_distance = distance
                closest_id = face_id
        
        if closest_id is not None:
            return closest_id
        else:
            # Create new face
            face_id = self.next_face_id
            self.next_face_id += 1
            return face_id
    
    def _cleanup_old_faces(self, current_time: float):
        """Remove faces that haven't been seen recently"""
        to_remove = []
        for face_id, face in self.tracked_faces.items():
            if current_time - face.last_seen > self.face_timeout:
                to_remove.append(face_id)
        
        for face_id in to_remove:
            del self.tracked_faces[face_id]
    
    def draw_faces(self, frame: np.ndarray, faces: List[Face]) -> np.ndarray:
        """Draw face bounding boxes and IDs on the frame"""
        for face in faces:
            x, y, w, h = face.bbox
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw face ID
            cv2.putText(frame, f"Person {face.id}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 0), 2)
            
            # Draw confidence
            cv2.putText(frame, f"{face.confidence:.2f}", 
                       (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (255, 255, 255), 1)
            
            # Draw center point
            cv2.circle(frame, face.center, 3, (0, 0, 255), -1)
        
        return frame
    
    def get_active_faces(self) -> List[Face]:
        """Get currently tracked faces"""
        return list(self.tracked_faces.values())
    
    def get_face_count(self) -> int:
        """Get number of currently tracked faces"""
        return len(self.tracked_faces)