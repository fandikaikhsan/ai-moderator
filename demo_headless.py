#!/usr/bin/env python3
"""
Quick demo of the AI Discussion Moderator headless features
Shows functionality even without camera for demonstration
"""

import time
import cv2
import numpy as np
from face_tracker import FaceTracker
from mouth_tracker import MouthTracker
from activity_tracker import ActivityTracker

def demo_ai_features():
    """Demonstrate AI features with simulated data"""
    print("🤖 AI Discussion Moderator - Feature Demo")
    print("="*50)
    
    # Initialize components
    face_tracker = FaceTracker()
    mouth_tracker = MouthTracker()
    activity_tracker = ActivityTracker()
    
    print("✅ All AI modules loaded successfully!")
    print()
    
    # Simulate discussion activity
    print("🎬 Simulating a 30-second discussion...")
    print("   (In real use, this analyzes live camera feed)")
    print()
    
    # Create sample faces data
    from face_tracker import Face
    
    # Simulate 3 participants with proper Face constructor
    current_time = time.time()
    participants = [
        Face(1, (100, 100, 80, 80), None, 0.95, current_time, (140, 140)),  # Person 1
        Face(2, (300, 150, 70, 70), None, 0.90, current_time, (335, 185)),  # Person 2  
        Face(3, (500, 200, 75, 75), None, 0.88, current_time, (537, 237)),  # Person 3
    ]
    
    print("👥 Detected participants: 3 people")
    print()
    
    # Simulate discussion over time
    simulation_steps = 10
    step_duration = 3  # seconds per step
    
    for step in range(simulation_steps):
        print(f"⏱️  Step {step + 1}/{simulation_steps} - Analyzing discussion...")
        
        # Simulate different speaking patterns
        from mouth_tracker import SpeechActivity
        speech_activities = {}
        current_time = time.time()
        
        if step < 3:
            # Person 1 dominates early
            speech_activities[1] = SpeechActivity(1, True, 0.8, 0.7, 15.0, 3.0, current_time)
            speech_activities[2] = SpeechActivity(2, False, 0.2, 0.1, 8.0, 0.5, current_time - 5)
            speech_activities[3] = SpeechActivity(3, False, 0.1, 0.05, 6.0, 0.2, current_time - 10)
        elif step < 6:
            # Person 2 takes over
            speech_activities[1] = SpeechActivity(1, False, 0.3, 0.2, 10.0, 1.0, current_time - 2)
            speech_activities[2] = SpeechActivity(2, True, 0.9, 0.8, 16.0, 4.0, current_time)
            speech_activities[3] = SpeechActivity(3, False, 0.2, 0.1, 7.0, 0.3, current_time - 8)
        elif step < 8:
            # Balanced discussion
            speech_activities[1] = SpeechActivity(1, True, 0.6, 0.5, 12.0, 2.5, current_time)
            speech_activities[2] = SpeechActivity(2, False, 0.4, 0.3, 11.0, 1.5, current_time - 1)
            speech_activities[3] = SpeechActivity(3, True, 0.7, 0.6, 13.0, 2.8, current_time)
        else:
            # Person 3 finally speaks up
            speech_activities[1] = SpeechActivity(1, False, 0.2, 0.1, 9.0, 0.8, current_time - 3)
            speech_activities[2] = SpeechActivity(2, False, 0.3, 0.2, 10.0, 1.2, current_time - 2)
            speech_activities[3] = SpeechActivity(3, True, 0.8, 0.7, 14.0, 3.5, current_time)
        
        # Update activity tracker
        activity_tracker.update_activity(speech_activities)
        
        # Show current status
        speaking_now = [pid for pid, activity in speech_activities.items() if activity.is_speaking]
        if speaking_now:
            print(f"   🗣️  Currently speaking: Person {', Person '.join(map(str, speaking_now))}")
        else:
            print("   🤐 Silence...")
        
        time.sleep(0.5)  # Short delay for demo
    
    print()
    print("📊 FINAL DISCUSSION ANALYSIS")
    print("="*50)
    
    # Get final statistics
    activity_matrix = activity_tracker.get_activity_matrix()
    
    if activity_matrix:
        # Sort by participation
        sorted_participants = sorted(activity_matrix.items(), 
                                   key=lambda x: x[1]['speaking_percentage'], 
                                   reverse=True)
        
        print("🏆 PARTICIPATION RANKING:")
        print("-" * 30)
        medals = ["🥇", "🥈", "🥉"]
        
        for i, (person_id, stats) in enumerate(sorted_participants):
            medal = medals[i] if i < 3 else f"{i+1}."
            level = stats['participation_level']
            percentage = stats['speaking_percentage']
            speaking_time = stats['speaking_time']
            
            print(f"{medal} Person {person_id}: {percentage:.1f}% ({speaking_time:.1f}s) - {level.upper()}")
        
        print()
        print("💡 MODERATOR INSIGHTS:")
        print("-" * 30)
        
        # Analysis
        dominant_count = sum(1 for stats in activity_matrix.values() if stats['participation_level'] == 'dominant')
        quiet_count = sum(1 for stats in activity_matrix.values() if stats['participation_level'] == 'quiet')
        
        if dominant_count > 1:
            print("• Multiple participants dominated - encourage turn-taking")
        elif dominant_count == 1:
            print("• One participant dominated - engage others more")
        
        if quiet_count > 0:
            print(f"• {quiet_count} participant(s) were quiet - check if they need support")
        
        if dominant_count == 0 and quiet_count <= 1:
            print("• Great discussion balance achieved! 🎉")
        
        # Show discussion pattern
        total_time = sum(stats['speaking_time'] for stats in activity_matrix.values())
        if total_time > 0:
            print(f"• Total active discussion: {total_time:.1f} seconds")
            
    print()
    print("✨ This demonstrates the core AI analysis features!")
    print("   In real use:")
    print("   📹 Live camera feed shows participants with face detection")
    print("   👄 Mouth movement analysis detects who's speaking")
    print("   📊 Real-time statistics update every 5 seconds")
    print("   🎯 Live participation balancing recommendations")

def test_camera_briefly():
    """Quick camera test"""
    print("\n🎥 Testing camera access...")
    
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ Camera is working!")
                print("   (In full mode, you'd see live video with face detection)")
            else:
                print("⚠️  Camera opened but couldn't read frame")
            cap.release()
        else:
            print("❌ Could not access camera")
    except Exception as e:
        print(f"❌ Camera error: {e}")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("🚀 Starting AI Discussion Moderator Demo...")
    print()
    
    demo_ai_features()
    test_camera_briefly()
    
    print()
    print("="*60)
    print("🎯 READY TO USE!")
    print("="*60)
    print("To use the full application:")
    print("1. Run: ./run.sh")
    print("2. Choose option 4 (Enhanced Headless Mode)")
    print("3. Position yourself in front of camera")
    print("4. Start your discussion!")
    print()
    print("Features you'll get:")
    print("✅ Real-time face detection and tracking")
    print("✅ Mouth movement analysis for speech detection")  
    print("✅ Live participation statistics every 5 seconds")
    print("✅ Discussion balance recommendations")
    print("✅ Final session report with rankings")
    print("="*60)