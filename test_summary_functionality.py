#!/usr/bin/env python3
"""
Test discussion summary functionality
"""

import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from discussion_summarizer import DiscussionSummarizer
from pyaudio_speech_transcriber import PyAudioSpeechTranscriber, TranscriptionSegment
from datetime import datetime

def test_summary_functionality():
    """Test the discussion summary functionality"""
    print("🧪 Testing Discussion Summary Functionality")
    print("=" * 50)
    
    # Test 1: Initialize summarizer
    print("\\n1. Testing Ollama integration...")
    try:
        summarizer = DiscussionSummarizer()
        print("✅ Discussion summarizer initialized")
    except Exception as e:
        print(f"❌ Failed to initialize summarizer: {e}")
        return
    
    # Test 2: Create mock transcript data
    print("\\n2. Creating mock transcript data...")
    mock_transcript = """[2025-09-13 15:10:00] Participant 1: Hello everyone, welcome to our project planning meeting.
[2025-09-13 15:10:05] Participant 2: Thanks for organizing this. I think we should discuss the timeline first.
[2025-09-13 15:10:10] Participant 1: Absolutely. We have three main deliverables to complete by next month.
[2025-09-13 15:10:15] Participant 2: Let's prioritize them based on client requirements.
[2025-09-13 15:10:20] Participant 1: I'll take the lead on the first deliverable. Can you handle the second one?
[2025-09-13 15:10:25] Participant 2: Sure, I can do that. We should also schedule a review meeting for next week."""
    
    mock_stats = {
        'duration_minutes': 5,
        'participant_count': 2,
        'total_segments': 6
    }
    
    print(f"📝 Mock transcript length: {len(mock_transcript)} characters")
    print(f"📊 Mock stats: {mock_stats}")
    
    # Test 3: Generate summary
    print("\\n3. Generating summary with Ollama...")
    try:
        summary = summarizer.generate_summary(mock_transcript, mock_stats)
        
        print("✅ Summary generated successfully!")
        print("\\n📋 SUMMARY RESULTS:")
        print("=" * 30)
        print(f"Summary: {summary.summary}")
        print(f"\\nKey Points: {summary.key_points}")
        print(f"\\nParticipants: {summary.participants_contribution}")
        print(f"\\nAction Items: {summary.action_items}")
        print(f"\\nSentiment: {summary.sentiment}")
        print(f"Duration: {summary.duration_minutes} minutes")
        
    except Exception as e:
        print(f"❌ Failed to generate summary: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Test transcriber integration
    print("\\n4. Testing transcriber integration...")
    try:
        transcriber = PyAudioSpeechTranscriber(device_index=None)
        
        # Add some mock segments
        seg1 = TranscriptionSegment(
            text="Hello everyone, let's start the meeting",
            timestamp=datetime.now(),
            participant_id="1",
            confidence=0.9
        )
        seg2 = TranscriptionSegment(
            text="I agree, we have a lot to discuss today",
            timestamp=datetime.now(),
            participant_id="2", 
            confidence=0.8
        )
        
        transcriber.transcription_history.append(seg1)
        transcriber.transcription_history.append(seg2)
        
        full_transcript = transcriber.get_full_transcript()
        stats = transcriber.get_discussion_stats()
        
        print(f"✅ Transcriber transcript: '{full_transcript}'")
        print(f"📊 Transcriber stats: {stats}")
        
        if full_transcript.strip():
            print("✅ Transcriber has content for summarization")
        else:
            print("⚠️ Transcriber transcript is empty")
            
    except Exception as e:
        print(f"❌ Transcriber test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_summary_functionality()