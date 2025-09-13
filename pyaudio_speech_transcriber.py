#!/usr/bin/env python3
"""
Alternative Speech Transcriber using direct PyAudio capture
Bypasses SpeechRecognition's problematic listen() method
"""

import pyaudio
import threading
import time
import queue
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from faster_whisper import WhisperModel
import logging
import wave
import io

@dataclass
class TranscriptionSegment:
    """Represents a segment of transcribed speech"""
    participant_id: int
    text: str
    timestamp: datetime
    confidence: float
    duration: float

class PyAudioSpeechTranscriber:
    """Alternative speech transcriber using direct PyAudio"""
    
    @staticmethod
    def get_available_microphones() -> List[Dict[str, any]]:
        """Get list of available microphone devices"""
        microphones = []
        
        try:
            p = pyaudio.PyAudio()
            
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                
                if device_info['maxInputChannels'] > 0:
                    microphones.append({
                        'index': i,
                        'name': device_info['name'],
                        'channels': device_info['maxInputChannels'],
                        'sample_rate': int(device_info['defaultSampleRate']),
                        'is_default': i == p.get_default_input_device_info()['index']
                    })
            
            p.terminate()
            
        except Exception as e:
            logging.error(f"Failed to enumerate microphones: {e}")
        
        return microphones
    
    def __init__(self, whisper_model_size: str = "base", microphone_index: Optional[int] = None):
        """Initialize the alternative speech transcriber"""
        self.whisper_model_size = whisper_model_size
        self.whisper_model = None
        self.microphone_index = microphone_index
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.audio_format = pyaudio.paInt16
        
        # Threading and queue management
        self.audio_queue = queue.Queue()
        self.transcription_thread = None
        self.is_listening = False
        
        # PyAudio objects
        self.pyaudio_instance = None
        self.audio_stream = None
        
        # Transcription storage
        self.transcription_history: List[TranscriptionSegment] = []
        self.current_transcriptions: Dict[int, str] = {}
        
        # Participant mapping
        self.face_to_participant = {}
        self.next_participant_id = 1
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self._load_whisper_model()
    
    def _load_whisper_model(self):
        """Load Whisper model for transcription"""
        try:
            self.logger.info(f"Loading Faster-Whisper model '{self.whisper_model_size}'...")
            self.whisper_model = WhisperModel(self.whisper_model_size, device="cpu", compute_type="int8")
            self.logger.info("Faster-Whisper model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load Faster-Whisper model: {e}")
            self.whisper_model = None
    
    def start_listening(self):
        """Start continuous audio listening using PyAudio directly"""
        if self.is_listening:
            return
        
        try:
            # Initialize PyAudio
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Open audio stream
            self.audio_stream = self.pyaudio_instance.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.microphone_index,
                frames_per_buffer=self.chunk_size
            )
            
            self.is_listening = True
            self.transcription_thread = threading.Thread(target=self._continuous_listen)
            self.transcription_thread.daemon = True
            self.transcription_thread.start()
            
            self.logger.info("Started PyAudio speech transcription")
            
        except Exception as e:
            self.logger.error(f"Failed to start audio listening: {e}")
            self._cleanup_audio()
    
    def stop_listening(self):
        """Stop audio listening"""
        self.is_listening = False
        
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.transcription_thread.join(timeout=2.0)
        
        self._cleanup_audio()
        self.logger.info("Stopped PyAudio speech transcription")
    
    def _cleanup_audio(self):
        """Clean up PyAudio resources"""
        try:
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None
            
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
                self.pyaudio_instance = None
                
        except Exception as e:
            self.logger.error(f"Error cleaning up audio: {e}")
    
    def _continuous_listen(self):
        """Continuous audio listening loop using direct PyAudio"""
        audio_buffer = []
        buffer_duration = 3.0  # Increased to 3 seconds for better efficiency
        samples_per_buffer = int(self.sample_rate * buffer_duration)
        
        try:
            chunk_count = 0
            while self.is_listening and self.audio_stream:
                try:
                    # Read audio data
                    data = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # Convert to numpy array
                    audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    audio_buffer.extend(audio_chunk)
                    
                    chunk_count += 1
                    
                    # Log audio levels less frequently for performance
                    if chunk_count % 50 == 0:  # Every 50 chunks instead of 10
                        rms = np.sqrt(np.mean(audio_chunk ** 2))
                        self.logger.debug(f"Audio chunk {chunk_count}: RMS={rms:.4f}")
                    
                    # Process when buffer is full
                    if len(audio_buffer) >= samples_per_buffer:
                        # Take samples and reset buffer
                        audio_data = np.array(audio_buffer[:samples_per_buffer])
                        audio_buffer = audio_buffer[samples_per_buffer//2:]  # 50% overlap
                        
                        # Check if audio has significant content (basic VAD)
                        if self._has_speech(audio_data):
                            # Only queue if we don't have too many pending items
                            if self.audio_queue.qsize() < 3:  # Limit queue size
                                self.logger.debug(f"Speech detected! Adding to queue (queue size: {self.audio_queue.qsize()})")
                                self.audio_queue.put({
                                    'audio': audio_data,
                                    'timestamp': datetime.now()
                                })
                            else:
                                self.logger.debug(f"Queue full, skipping audio segment")
                        
                except Exception as e:
                    self.logger.error(f"Error in PyAudio listening: {type(e).__name__}: {e}")
                    time.sleep(0.1)
                    
        except Exception as e:
            self.logger.error(f"Fatal error in PyAudio listening: {type(e).__name__}: {e}")
    
    def _has_speech(self, audio_data: np.ndarray) -> bool:
        """Simple Voice Activity Detection"""
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_data ** 2))
        
        # Adjusted threshold - slightly higher to reduce false positives
        threshold = 0.002  # Slightly higher threshold
        has_speech = rms > threshold
        
        # Only log when speech is detected to reduce logging overhead
        if has_speech:
            self.logger.debug(f"VAD: Speech detected (RMS: {rms:.4f} > {threshold})")
        
        return has_speech
    
    def process_audio_queue(self, active_faces: List[int]) -> Dict[int, str]:
        """Process queued audio and return new transcriptions"""
        new_transcriptions = {}
        
        # Process all queued audio
        while not self.audio_queue.empty():
            try:
                audio_item = self.audio_queue.get_nowait()
                transcript = self._transcribe_audio(audio_item['audio'])
                
                if transcript.strip():
                    # Assign to participant (simple assignment to first active face)
                    participant_id = active_faces[0] if active_faces else 1
                    
                    # Create transcription segment
                    segment = TranscriptionSegment(
                        participant_id=participant_id,
                        text=transcript,
                        timestamp=audio_item['timestamp'],
                        confidence=0.8,  # Placeholder confidence
                        duration=2.0
                    )
                    
                    self.transcription_history.append(segment)
                    new_transcriptions[participant_id] = transcript
                    
                    self.logger.info(f"Transcribed: {transcript}")
                    
            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"Error processing audio: {e}")
        
        return new_transcriptions
    
    def _transcribe_audio(self, audio_data: np.ndarray) -> str:
        """Transcribe audio using Faster-Whisper"""
        if not self.whisper_model:
            return ""
        
        try:
            # Whisper expects audio at 16kHz
            if len(audio_data) == 0:
                return ""
            
            # Transcribe with Faster-Whisper
            segments, info = self.whisper_model.transcribe(
                audio_data,
                beam_size=5,
                language='en'
            )
            
            # Combine all segments
            transcript = " ".join([segment.text.strip() for segment in segments])
            return transcript.strip()
            
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            return ""
    
    def change_microphone(self, microphone_index: Optional[int]) -> bool:
        """Change the microphone device"""
        was_listening = self.is_listening
        
        if was_listening:
            self.stop_listening()
        
        self.microphone_index = microphone_index
        
        if was_listening:
            self.start_listening()
        
        self.logger.info(f"Changed microphone to device {microphone_index}")
        return True
    
    def get_current_microphone_info(self) -> Dict[str, any]:
        """Get information about currently selected microphone"""
        microphones = self.get_available_microphones()
        
        if self.microphone_index is not None:
            for mic in microphones:
                if mic['index'] == self.microphone_index:
                    return mic
        
        for mic in microphones:
            if mic.get('is_default', False):
                return mic
        
        return microphones[0] if microphones else {
            'index': None,
            'name': 'Unknown',
            'channels': 1,
            'sample_rate': 16000,
            'is_default': True
        }
    
    def clear_transcriptions(self):
        """Clear all transcription data"""
        self.transcription_history.clear()
        self.current_transcriptions.clear()
        
        # Clear audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        self.logger.info("Cleared all transcription data")
    
    def get_full_transcript(self) -> str:
        """Get full transcript of the discussion"""
        return " ".join([segment.text for segment in self.transcription_history])
    
    def get_discussion_stats(self) -> Dict:
        """Get basic discussion statistics"""
        return {
            'total_segments': len(self.transcription_history),
            'participants': len(set(segment.participant_id for segment in self.transcription_history))
        }