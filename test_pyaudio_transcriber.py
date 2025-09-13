#!/usr/bin/env python3
"""
Test the alternative PyAudio-based speech transcriber
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pyaudio_speech_transcriber import PyAudioSpeechTranscriber
import time

def test_pyaudio_transcriber():
    """Test the PyAudio-based transcriber"""
    print("🎤 Testing PyAudio Speech Transcriber")
    print("=" * 40)
    
    # Get available microphones
    microphones = PyAudioSpeechTranscriber.get_available_microphones()
    
    print("Available microphones:")
    for i, mic in enumerate(microphones):
        default_marker = " (DEFAULT)" if mic.get('is_default', False) else ""
        print(f"  {i+1}. Index {mic['index']}: {mic['name']}{default_marker}")
    
    print()
    
    # Test with default microphone
    default_mic = None
    for mic in microphones:
        if mic.get('is_default', False):
            default_mic = mic
            break
    
    if not default_mic:
        default_mic = microphones[0] if microphones else None
    
    if not default_mic:
        print("❌ No microphones available")
        return False
    
    print(f"🔧 Testing with: {default_mic['name']} (Index: {default_mic['index']})")
    print("-" * 50)
    
    try:
        # Create transcriber
        transcriber = PyAudioSpeechTranscriber(microphone_index=default_mic['index'])
        print("✅ Transcriber created successfully")
        
        # Start listening
        print("🎯 Starting transcription...")
        transcriber.start_listening()
        print("⏱️ Listening for 10 seconds... Please speak something!")
        
        # Monitor for audio capture
        start_time = time.time()
        last_queue_size = 0
        
        while time.time() - start_time < 10:
            time.sleep(1)
            
            # Check audio queue
            current_queue_size = transcriber.audio_queue.qsize()
            if current_queue_size > last_queue_size:
                print(f"   📊 Audio chunks captured: {current_queue_size}")
                last_queue_size = current_queue_size
            
            # Process any queued audio
            new_transcriptions = transcriber.process_audio_queue([1])
            for participant_id, text in new_transcriptions.items():
                print(f"   🗣️ Participant {participant_id}: {text}")
        
        # Stop listening
        transcriber.stop_listening()
        print("⏹️ Stopped transcription")
        
        # Final stats
        total_captured = transcriber.audio_queue.qsize()
        full_transcript = transcriber.get_full_transcript()
        
        print(f"📊 Final Results:")
        print(f"   Audio chunks captured: {total_captured}")
        print(f"   Total transcript length: {len(full_transcript)} characters")
        
        if full_transcript.strip():
            print(f"   Full transcript: {full_transcript}")
            print("✅ SUCCESS: PyAudio transcriber working!")
            return True
        else:
            print("⚠️ No speech transcribed (but audio was captured)")
            return total_captured > 0
            
    except Exception as e:
        print(f"❌ FAILED: Error with PyAudio transcriber: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 PyAudio Speech Transcriber Test")
    print("=" * 50)
    
    try:
        success = test_pyaudio_transcriber()
        
        print("\\n🎯 TEST SUMMARY")
        print("=" * 20)
        if success:
            print("✅ PyAudio transcriber working!")
            print("💡 This can replace the problematic SpeechRecognition version")
        else:
            print("❌ PyAudio transcriber failed")
            print("💡 Check microphone permissions and hardware")
            
    except KeyboardInterrupt:
        print("\\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\\n❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()