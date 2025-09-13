#!/usr/bin/env python3
"""
Debug speech transcription audio input issues
"""

import speech_recognition as sr
import pyaudio
import time
import threading
import queue

def test_microphone_access():
    """Test basic microphone access"""
    print("🎤 Testing Microphone Access...")
    print("=" * 40)
    
    try:
        # List available microphones
        print("Available microphones:")
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"  {index}: {name}")
        
        # Test default microphone
        r = sr.Recognizer()
        mic = sr.Microphone()
        
        print(f"\n🎯 Using default microphone")
        print("📊 Adjusting for ambient noise (2 seconds)...")
        
        with mic as source:
            r.adjust_for_ambient_noise(source, duration=2)
            
        print(f"✅ Energy threshold: {r.energy_threshold}")
        print(f"✅ Pause threshold: {r.pause_threshold}")
        
        return True, r, mic
        
    except Exception as e:
        print(f"❌ Microphone access failed: {e}")
        return False, None, None

def test_audio_recording():
    """Test basic audio recording"""
    print("\n🔊 Testing Audio Recording...")
    print("=" * 40)
    
    success, recognizer, microphone = test_microphone_access()
    if not success:
        return False
    
    try:
        print("🎙️  Speak now for 3 seconds...")
        
        with microphone as source:
            audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
            
        print(f"✅ Audio captured: {len(audio.get_raw_data())} bytes")
        print(f"✅ Sample rate: {audio.sample_rate}")
        print(f"✅ Sample width: {audio.sample_width}")
        
        return True
        
    except sr.WaitTimeoutError:
        print("⚠️  No audio detected (timeout)")
        return False
    except Exception as e:
        print(f"❌ Audio recording failed: {e}")
        return False

def test_google_recognition():
    """Test Google speech recognition (requires internet)"""
    print("\n🧠 Testing Google Speech Recognition...")
    print("=" * 40)
    
    success, recognizer, microphone = test_microphone_access()
    if not success:
        return False
    
    try:
        print("🎙️  Speak clearly for 3 seconds...")
        
        with microphone as source:
            audio = recognizer.listen(source, timeout=2, phrase_time_limit=3)
            
        print("🔄 Processing with Google Speech Recognition...")
        text = recognizer.recognize_google(audio)
        
        print(f"✅ Recognized: '{text}'")
        return True
        
    except sr.UnknownValueError:
        print("⚠️  Could not understand audio")
        return False
    except sr.RequestError as e:
        print(f"❌ Google API error: {e}")
        return False
    except sr.WaitTimeoutError:
        print("⚠️  No audio detected (timeout)")
        return False
    except Exception as e:
        print(f"❌ Recognition failed: {e}")
        return False

def test_continuous_listening():
    """Test continuous listening like our app does"""
    print("\n🔄 Testing Continuous Listening...")
    print("=" * 40)
    
    success, recognizer, microphone = test_microphone_access()
    if not success:
        return False
    
    audio_queue = queue.Queue()
    listening = True
    
    def listen_continuously():
        """Continuous listening thread"""
        try:
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source)
                
            while listening:
                try:
                    with microphone as source:
                        audio = recognizer.listen(source, timeout=1, phrase_time_limit=2)
                        audio_queue.put(audio)
                        print("🎤 Audio captured and queued")
                        
                except sr.WaitTimeoutError:
                    pass  # Normal timeout
                except Exception as e:
                    print(f"❌ Listen error: {e}")
                    
        except Exception as e:
            print(f"❌ Continuous listening failed: {e}")
    
    # Start listening thread
    listen_thread = threading.Thread(target=listen_continuously)
    listen_thread.daemon = True
    listen_thread.start()
    
    print("🎙️  Speak multiple times over the next 10 seconds...")
    print("     (The system will capture audio continuously)")
    
    start_time = time.time()
    audio_count = 0
    
    while time.time() - start_time < 10:
        try:
            audio = audio_queue.get(timeout=0.5)
            audio_count += 1
            
            # Try to recognize each audio
            try:
                text = recognizer.recognize_google(audio)
                print(f"✅ #{audio_count}: '{text}'")
            except:
                print(f"⚠️  #{audio_count}: Audio captured but not recognized")
                
        except queue.Empty:
            pass
    
    listening = False
    
    print(f"\n📊 Results: {audio_count} audio segments captured")
    return audio_count > 0

def test_whisper_model():
    """Test if Whisper model can be loaded"""
    print("\n🤖 Testing Whisper Model...")
    print("=" * 40)
    
    try:
        from faster_whisper import WhisperModel
        
        print("📥 Loading Whisper 'tiny' model...")
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("✅ Whisper model loaded successfully")
        
        # Test with sample audio if we have any
        print("💡 Whisper model is ready for transcription")
        return True
        
    except Exception as e:
        print(f"❌ Whisper model failed: {e}")
        return False

def main():
    print("🧪 Speech Transcription Debug Tool")
    print("=" * 50)
    
    # Test 1: Microphone access
    mic_success, _, _ = test_microphone_access()
    
    # Test 2: Audio recording
    audio_success = test_audio_recording() if mic_success else False
    
    # Test 3: Google recognition
    google_success = test_google_recognition() if audio_success else False
    
    # Test 4: Continuous listening
    continuous_success = test_continuous_listening() if mic_success else False
    
    # Test 5: Whisper model
    whisper_success = test_whisper_model()
    
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC RESULTS:")
    print("=" * 50)
    print(f"  Microphone Access:     {'✅ Working' if mic_success else '❌ Failed'}")
    print(f"  Audio Recording:       {'✅ Working' if audio_success else '❌ Failed'}")
    print(f"  Google Recognition:    {'✅ Working' if google_success else '❌ Failed'}")
    print(f"  Continuous Listening:  {'✅ Working' if continuous_success else '❌ Failed'}")
    print(f"  Whisper Model:         {'✅ Working' if whisper_success else '❌ Failed'}")
    
    if all([mic_success, audio_success, continuous_success, whisper_success]):
        print("\n🎉 All systems working! The issue might be in the app logic.")
        print("💡 Check:")
        print("   • Transcription thread is actually started")
        print("   • Audio queue processing loop")
        print("   • GUI callback connections")
    else:
        print("\n⚠️  Found issues that need to be fixed:")
        if not mic_success:
            print("   • Check microphone permissions")
        if not audio_success:
            print("   • Check audio input device")
        if not google_success:
            print("   • Check internet connection for Google API")
        if not continuous_success:
            print("   • Check continuous listening implementation")
        if not whisper_success:
            print("   • Check Whisper model installation")

if __name__ == "__main__":
    main()