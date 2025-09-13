#!/usr/bin/env python3
"""
Enhanced Audio Debug Tool - Detailed audio pipeline testing
"""

import speech_recognition as sr
import pyaudio
import time
import threading
import queue
from datetime import datetime
import traceback
import sys

def test_microphone_access():
    """Test basic microphone access"""
    print("🎤 Testing microphone access...")
    
    try:
        # Test PyAudio directly
        p = pyaudio.PyAudio()
        
        # List all audio devices
        print(f"📱 Found {p.get_device_count()} audio devices:")
        default_input = None
        
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                print(f"   Device {i}: {device_info['name']} (Inputs: {device_info['maxInputChannels']})")
                if default_input is None:
                    default_input = i
        
        if default_input is None:
            print("❌ No input devices found!")
            return False
        
        print(f"🎯 Using device {default_input} as default input")
        
        # Test opening input stream
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                input_device_index=default_input,
                frames_per_buffer=1024
            )
            
            print("✅ Successfully opened input stream")
            
            # Test reading some audio data
            print("📊 Testing audio input (2 seconds)...")
            for i in range(10):  # 10 chunks * 0.2s = 2s
                data = stream.read(1024, exception_on_overflow=False)
                print(f"   Chunk {i+1}: Read {len(data)} bytes")
                time.sleep(0.2)
            
            stream.stop_stream()
            stream.close()
            print("✅ Audio input test successful")
            
        except Exception as e:
            print(f"❌ Failed to open input stream: {e}")
            print(f"   Full error: {traceback.format_exc()}")
            return False
        finally:
            p.terminate()
        
        return True
        
    except Exception as e:
        print(f"❌ PyAudio initialization failed: {e}")
        print(f"   Full error: {traceback.format_exc()}")
        return False

def test_speech_recognition_basic():
    """Test basic SpeechRecognition setup"""
    print("\\n🗣️ Testing SpeechRecognition basic setup...")
    
    try:
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        
        print("✅ SpeechRecognition objects created successfully")
        
        # Test microphone access
        try:
            with microphone as source:
                print("📊 Adjusting for ambient noise (this may take a moment)...")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                print(f"   Energy threshold set to: {recognizer.energy_threshold}")
                
            print("✅ Microphone access and ambient noise adjustment successful")
            return True
            
        except Exception as e:
            print(f"❌ Microphone access failed: {e}")
            print(f"   Type: {type(e).__name__}")
            print(f"   Full error: {traceback.format_exc()}")
            return False
            
    except Exception as e:
        print(f"❌ SpeechRecognition setup failed: {e}")
        print(f"   Full error: {traceback.format_exc()}")
        return False

def test_continuous_listening():
    """Test the exact continuous listening pattern used in the app"""
    print("\\n🔄 Testing continuous listening pattern...")
    
    try:
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        
        # Configure like in the app
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.3
        
        audio_queue = queue.Queue()
        is_listening = True
        
        def continuous_listen():
            """Exact copy of the problematic function"""
            try:
                with microphone as source:
                    print(f"🎯 Starting continuous listening loop...")
                    loop_count = 0
                    
                    while is_listening and loop_count < 50:  # Limit for testing
                        try:
                            loop_count += 1
                            print(f"   Loop {loop_count}: Listening...")
                            
                            # This is the exact call that's failing
                            audio_data = recognizer.listen(
                                source, 
                                timeout=1.0,
                                phrase_time_limit=5.0
                            )
                            
                            print(f"   Loop {loop_count}: Got audio data ({len(audio_data.get_raw_data())} bytes)")
                            
                            # Add to queue
                            audio_queue.put({
                                'audio': audio_data,
                                'timestamp': datetime.now()
                            })
                            
                        except sr.WaitTimeoutError:
                            print(f"   Loop {loop_count}: Timeout (normal)")
                            pass
                        except Exception as e:
                            print(f"❌ Loop {loop_count}: Error in listening: {repr(e)}")
                            print(f"   Type: {type(e).__name__}")
                            print(f"   Args: {e.args}")
                            print(f"   Str: '{str(e)}'")
                            if str(e).strip() == '':
                                print("   ⚠️ Empty error string detected!")
                            time.sleep(0.1)
                            
            except Exception as e:
                print(f"❌ Fatal error in continuous listening: {repr(e)}")
                print(f"   Type: {type(e).__name__}")
                print(f"   Full error: {traceback.format_exc()}")
        
        # Start listening thread
        thread = threading.Thread(target=continuous_listen)
        thread.daemon = True
        thread.start()
        
        print("⏱️ Running for 10 seconds...")
        time.sleep(10)
        
        is_listening = False
        thread.join(timeout=2)
        
        print(f"📊 Results: {audio_queue.qsize()} audio items captured")
        
        return True
        
    except Exception as e:
        print(f"❌ Continuous listening test failed: {e}")
        print(f"   Full error: {traceback.format_exc()}")
        return False

def test_microphone_permissions():
    """Test microphone permissions on macOS"""
    print("\\n🔐 Testing microphone permissions...")
    
    try:
        import subprocess
        
        # Check microphone permissions (macOS specific)
        try:
            result = subprocess.run(['tccutil', 'reset', 'Microphone'], 
                                  capture_output=True, text=True, timeout=5)
            print("💡 You might need to grant microphone permissions")
            print("   Go to System Preferences > Security & Privacy > Privacy > Microphone")
            print("   and ensure Python/Terminal has access")
        except:
            pass
        
        return True
        
    except Exception as e:
        print(f"❌ Permission check failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Enhanced Audio Debug Tool")
    print("=" * 50)
    
    tests = [
        ("Microphone Access", test_microphone_access),
        ("SpeechRecognition Basic", test_speech_recognition_basic), 
        ("Microphone Permissions", test_microphone_permissions),
        ("Continuous Listening", test_continuous_listening),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            results[test_name] = test_func()
        except KeyboardInterrupt:
            print("\\n⚠️ Test interrupted by user")
            break
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            print(f"   Full error: {traceback.format_exc()}")
            results[test_name] = False
    
    # Summary
    print(f"\\n{'=' * 50}")
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    print(f"\\n🎯 {passed_tests}/{total_tests} tests passed")
    
    if passed_tests < total_tests:
        print("\\n💡 TROUBLESHOOTING TIPS:")
        print("1. Grant microphone permissions to Terminal/Python")
        print("2. Check if other apps are using the microphone")
        print("3. Try restarting the Terminal/Python")
        print("4. Check audio input device in System Preferences")

if __name__ == "__main__":
    main()