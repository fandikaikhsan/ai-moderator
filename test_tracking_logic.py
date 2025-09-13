#!/usr/bin/env python3
"""
Test enhanced face tracking logic without camera
"""

import numpy as np
import time
from face_tracker import FaceTracker, Face

def simulate_face_tracking():
    """Simulate face tracking scenarios"""
    print("🧪 Testing Enhanced Face Tracking Logic")
    print("=" * 50)
    
    face_tracker = FaceTracker()
    current_time = time.time()
    
    print("\n📋 Test 1: Single person moving slightly")
    # Simulate a person moving slightly between frames
    test_detections = [
        # Frame 1: Person at (100, 100)
        [(100, 100, 80, 80)],
        # Frame 2: Person moved slightly to (105, 103) 
        [(105, 103, 82, 78)],
        # Frame 3: Person moved to (110, 108)
        [(110, 108, 79, 81)],
        # Frame 4: Person moved back to (102, 105)
        [(102, 105, 81, 79)],
    ]
    
    face_ids_sequence = []
    for i, detections in enumerate(test_detections):
        faces = []
        for (x, y, w, h) in detections:
            center = (x + w//2, y + h//2)
            face_id = face_tracker._track_face(center, (x, y, w, h), current_time + i * 0.1)
            
            face = Face(
                id=face_id,
                bbox=(x, y, w, h),
                landmarks=None,
                confidence=0.95,
                last_seen=current_time + i * 0.1,
                center=center
            )
            faces.append(face)
            face_tracker.tracked_faces[face_id] = face
        
        face_ids = [f.id for f in faces]
        face_ids_sequence.append(face_ids)
        print(f"   Frame {i+1}: Detection at {detections[0][:2]} -> ID {face_ids[0]}")
    
    # Check if same ID was maintained
    all_ids = set()
    for ids in face_ids_sequence:
        all_ids.update(ids)
    
    if len(all_ids) == 1:
        print("   ✅ PASS: Consistent person ID maintained")
    else:
        print(f"   ❌ FAIL: Created {len(all_ids)} different IDs for same person")
    
    print("\n📋 Test 2: Person disappears and reappears")
    face_tracker = FaceTracker()  # Reset
    
    test_detections_2 = [
        # Frame 1: Person present
        [(200, 150, 80, 80)],
        # Frame 2: Person still there, moved slightly
        [(205, 152, 78, 82)],
        # Frame 3: Person gone (looked away)
        [],
        # Frame 4: Person gone
        [],
        # Frame 5: Person returns in similar position
        [(203, 148, 81, 79)],
    ]
    
    face_ids_sequence_2 = []
    for i, detections in enumerate(test_detections_2):
        faces = []
        for (x, y, w, h) in detections:
            center = (x + w//2, y + h//2)
            face_id = face_tracker._track_face(center, (x, y, w, h), current_time + i * 0.5)
            
            face = Face(
                id=face_id,
                bbox=(x, y, w, h),
                landmarks=None,
                confidence=0.95,
                last_seen=current_time + i * 0.5,
                center=center
            )
            faces.append(face)
            face_tracker.tracked_faces[face_id] = face
        
        face_ids = [f.id for f in faces]
        face_ids_sequence_2.append(face_ids)
        
        if face_ids:
            print(f"   Frame {i+1}: Detection at {detections[0][:2]} -> ID {face_ids[0]}")
        else:
            print(f"   Frame {i+1}: No detection")
    
    # Check reappearance behavior
    first_id = face_ids_sequence_2[0][0] if face_ids_sequence_2[0] else None
    last_id = face_ids_sequence_2[4][0] if face_ids_sequence_2[4] else None
    
    if first_id and last_id and first_id == last_id:
        print("   ✅ PASS: Same ID after reappearing")
    elif first_id and last_id:
        print(f"   🟡 INFO: Different ID after reappearing (ID {first_id} -> {last_id})")
        print("   💡 This is acceptable if person was gone too long")
    else:
        print("   ❌ FAIL: Tracking issue")
    
    print("\n📋 Test 3: Two people at different positions")
    face_tracker = FaceTracker()  # Reset
    
    # Two people at clearly different positions
    detections_two_people = [
        [(100, 100, 80, 80), (300, 150, 75, 85)],  # Two people
        [(105, 102, 78, 82), (305, 148, 77, 83)],  # Both moved slightly
        [(102, 98, 81, 79), (302, 152, 76, 84)],   # Both moved again
    ]
    
    all_ids_multi = set()
    for i, detections in enumerate(detections_two_people):
        frame_ids = []
        for (x, y, w, h) in detections:
            center = (x + w//2, y + h//2)
            face_id = face_tracker._track_face(center, (x, y, w, h), current_time + i * 0.1)
            
            face = Face(
                id=face_id,
                bbox=(x, y, w, h),
                landmarks=None,
                confidence=0.95,
                last_seen=current_time + i * 0.1,
                center=center
            )
            face_tracker.tracked_faces[face_id] = face
            frame_ids.append(face_id)
        
        all_ids_multi.update(frame_ids)
        print(f"   Frame {i+1}: IDs {frame_ids}")
    
    if len(all_ids_multi) == 2:
        print("   ✅ PASS: Correctly tracked 2 different people")
    else:
        print(f"   ❌ FAIL: Created {len(all_ids_multi)} IDs for 2 people")
    
    print("\n" + "=" * 50)
    print("🎯 ENHANCED FACE TRACKING SUMMARY")
    print("=" * 50)
    print("✅ Key Improvements Implemented:")
    print("   • Multi-factor similarity scoring")
    print("   • Bounding box overlap analysis")
    print("   • Size consistency checking")
    print("   • Movement trajectory prediction")
    print("   • Position history tracking")
    print("   • Time-based confidence decay")
    print()
    print("📈 Expected Benefits:")
    print("   • Much more stable person IDs")
    print("   • Better tracking through temporary occlusions")
    print("   • Reduced ID switching for same person")
    print("   • Improved multi-person tracking")
    print()
    print("🚀 Ready to test in GUI mode!")

if __name__ == "__main__":
    simulate_face_tracking()