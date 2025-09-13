#!/usr/bin/env python3
"""
Test microphone selection functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from speech_transcriber import SpeechTranscriber
import time

def test_microphone_selection():
    """Test microphone selection and transcription"""
    print("🎤 Testing Microphone Selection")
    print("=" * 40)
    
    # Get available microphones
    microphones = SpeechTranscriber.get_available_microphones()
    
    print("Available microphones:")
    for i, mic in enumerate(microphones):
        default_marker = " (DEFAULT)" if mic.get('is_default', False) else ""
        print(f"  {i+1}. Index {mic['index']}: {mic['name']}{default_marker}")
        print(f"     Channels: {mic['channels']}, Sample Rate: {mic['sample_rate']} Hz")
    
    print()
    
    # Test different microphones
    for mic in microphones[:2]:  # Test first 2 microphones
        print(f"\\n🔧 Testing microphone: {mic['name']} (Index: {mic['index']})")
        print("-" * 50)
        
        try:
            # Create transcriber with specific microphone
            transcriber = SpeechTranscriber(microphone_index=mic['index'])
            
            print("✅ Transcriber created successfully")
            
            # Test starting listening
            print("🎯 Starting transcription...")
            transcriber.start_listening()
            
            print("⏱️ Testing for 5 seconds... (speak something)")
            time.sleep(5)
            
            # Stop and check results
            transcriber.stop_listening()
            print("⏹️ Stopped transcription")
            
            # Check if any audio was captured
            queue_size = transcriber.audio_queue.qsize()
            print(f"📊 Audio items captured: {queue_size}")
            
            if queue_size > 0:
                print(f"✅ SUCCESS: Microphone {mic['name']} captured audio!")
                return True
            else:
                print(f"⚠️ No audio captured with {mic['name']}")
                
        except Exception as e:
            print(f"❌ FAILED: Error with microphone {mic['name']}: {e}")
            print(f"   Error type: {type(e).__name__}")
    
    return False

def test_change_microphone():
    """Test dynamic microphone changing"""
    print("\\n🔄 Testing Dynamic Microphone Change")
    print("=" * 40)
    
    microphones = SpeechTranscriber.get_available_microphones()
    
    if len(microphones) < 2:
        print("⚠️ Need at least 2 microphones to test changing")
        return
    
    try:
        # Start with default microphone
        transcriber = SpeechTranscriber()
        print(f"🎯 Started with default microphone")
        
        # Change to different microphone
        new_mic = microphones[1]  # Use second microphone
        success = transcriber.change_microphone(new_mic['index'])
        
        if success:
            print(f"✅ Successfully changed to: {new_mic['name']}")
        else:
            print(f"❌ Failed to change to: {new_mic['name']}")
            
        # Get current microphone info
        current_info = transcriber.get_current_microphone_info()
        print(f"📋 Current microphone: {current_info['name']}")
        
    except Exception as e:
        print(f"❌ Error testing microphone change: {e}")

if __name__ == "__main__":
    print("🧪 Microphone Selection Test Suite")
    print("=" * 50)
    
    try:
        # Test 1: Basic microphone selection
        success = test_microphone_selection()
        
        # Test 2: Dynamic microphone change
        test_change_microphone()
        
        print("\\n🎯 TEST SUMMARY")
        print("=" * 20)
        if success:
            print("✅ At least one microphone working!")
            print("💡 Try using that microphone in the main application")
        else:
            print("❌ No microphones captured audio successfully")
            print("💡 This might be a permissions or hardware issue")
            
    except KeyboardInterrupt:
        print("\\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\\n❌ Test suite error: {e}")