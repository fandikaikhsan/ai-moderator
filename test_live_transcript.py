#!/usr/bin/env python3
"""
Test the actual AI moderator with transcript export functionality
"""
import time
import signal
import sys
import subprocess

def signal_handler(sig, frame):
    print('\nStopping test...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

print("Starting AI Moderator for 15 seconds with transcript export...")
print("The system will capture video from your camera and track speaking patterns.")
print("Move your mouth to simulate speaking and test the transcript functionality.")
print("Press Ctrl+C to stop early.")

# Start the AI moderator with export functionality
try:
    process = subprocess.Popen([
        'python', 'main.py', '--no-gui', '--export', 'live_test_transcript.json'
    ], cwd='/Users/930341/Library/CloudStorage/OneDrive-Personal/Documents/CMU/HackCMU/ai-moderator')
    
    # Wait for 15 seconds
    time.sleep(15)
    
    # Stop the process
    process.terminate()
    process.wait()
    
    print("\nTest completed! Check live_test_transcript.json for results.")
    
except KeyboardInterrupt:
    print("\nTest interrupted by user.")
    if 'process' in locals():
        process.terminate()
        process.wait()
except Exception as e:
    print(f"Error during test: {e}")
    if 'process' in locals():
        process.terminate()
        process.wait()