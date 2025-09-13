#!/usr/bin/env python3
"""
Debug transcription data and summary generation
"""

import sys
import os
from datetime import datetime

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pyaudio_speech_transcriber import PyAudioSpeechTranscriber, TranscriptionSegment
from lightweight_summarizer import LightweightSummarizer

def debug_transcription_and_summary():
    """Debug what's happening with transcription and summary"""
    print("🔍 Debugging Transcription and Summary")
    print("=" * 50)
    
    # Create transcriber and summarizer
    transcriber = PyAudioSpeechTranscriber(device_index=None)
    summarizer = LightweightSummarizer()
    
    # Add some realistic transcription segments (like what we saw in the logs)
    print("\\n1. Adding mock transcription segments...")
    segments = [
        TranscriptionSegment(
            text="Hello everyone, hello everyone, welcome to India",
            timestamp=datetime.now(),
            participant_id="1",
            confidence=0.9
        ),
        TranscriptionSegment(
            text="Hello everyone, my name is Andika",
            timestamp=datetime.now(),
            participant_id="1", 
            confidence=0.85
        ),
        TranscriptionSegment(
            text="Thank you, Anika",
            timestamp=datetime.now(),
            participant_id="2",
            confidence=0.8
        ),
        TranscriptionSegment(
            text="I want to take a mobile first",
            timestamp=datetime.now(),
            participant_id="1",
            confidence=0.75
        ),
        TranscriptionSegment(
            text="want to take a boba first",
            timestamp=datetime.now(),
            participant_id="2",
            confidence=0.7
        )
    ]
    
    # Add segments to transcriber
    for segment in segments:
        transcriber.transcription_history.append(segment)
    
    print(f"✅ Added {len(segments)} transcription segments")
    
    # Check what the transcriber returns
    print("\\n2. Checking transcriber output...")
    full_transcript = transcriber.get_full_transcript()
    discussion_stats = transcriber.get_discussion_stats()
    
    print(f"📝 Full transcript:")
    print(f"'{full_transcript}'")
    print(f"\\n📊 Discussion stats:")
    print(f"{discussion_stats}")
    
    # Test the summarizer
    print("\\n3. Testing summarizer...")
    if full_transcript.strip():
        print("✅ Transcript has content, generating summary...")
        summary = summarizer.generate_summary(full_transcript, discussion_stats)
        
        print("\\n📋 SUMMARY RESULTS:")
        print("=" * 30)
        print(f"Summary: {summary.summary}")
        print(f"\\nKey Points: {summary.key_points}")
        print(f"\\nParticipants: {summary.participants_contribution}")
        print(f"\\nAction Items: {summary.action_items}")
        print(f"\\nSentiment: {summary.sentiment}")
        print(f"Duration: {summary.duration_minutes} minutes")
    else:
        print("❌ Transcript is empty!")
    
    # Debug the parsing
    print("\\n4. Debugging transcript parsing...")
    lines = full_transcript.strip().split('\\n')
    print(f"Transcript split into {len(lines)} lines:")
    for i, line in enumerate(lines):
        print(f"  Line {i}: '{line}'")
    
    # Test the parsing function directly
    print("\\n5. Testing parsing function...")
    segments_parsed = summarizer._parse_transcript(full_transcript)
    print(f"Parsed {len(segments_parsed)} segments:")
    for i, seg in enumerate(segments_parsed):
        print(f"  Segment {i}: {seg}")

def test_with_gui_callback():
    """Test what happens when the GUI callback is triggered"""
    print("\\n\\n🎯 Testing GUI Summary Callback")
    print("=" * 40)
    
    # Simulate what happens in main.py
    from main import AIModerator
    
    # Create moderator (but don't start it)
    moderator = AIModerator(gui_mode=False)
    
    # Add some mock data to the speech transcriber
    segments = [
        TranscriptionSegment(
            text="Hello everyone, welcome to our meeting",
            timestamp=datetime.now(),
            participant_id="1",
            confidence=0.9
        ),
        TranscriptionSegment(
            text="Thanks for organizing this discussion",
            timestamp=datetime.now(),
            participant_id="2",
            confidence=0.85
        )
    ]
    
    for segment in segments:
        moderator.speech_transcriber.transcription_history.append(segment)
    
    # Test the summary generation method
    print("Testing _generate_discussion_summary method...")
    try:
        moderator._generate_discussion_summary()
        print("✅ Summary generation completed successfully")
    except Exception as e:
        print(f"❌ Summary generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_transcription_and_summary()
    test_with_gui_callback()