"""
Speech Transcription Module for AI Discussion Moderator
Handles real-time audio capture, speech recognition, and participant mapping
"""

import speech_recognition as sr
import threading
import time
import queue
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from faster_whisper import WhisperModel
import logging
import pyaudio

@dataclass
class TranscriptionSegment:
    """Represents a segment of transcribed speech"""
    participant_id: int
    text: str
    timestamp: datetime
    confidence: float
    duration: float

class SpeechTranscriber:
    """Handles real-time speech transcription with participant identification"""
    
    @staticmethod
    def get_available_microphones() -> List[Dict[str, any]]:
        """
        Get list of available microphone devices
        
        Returns:
            List of dictionaries containing device info (index, name, channels)
        """
        microphones = []
        
        try:
            # Use PyAudio directly for more detailed device info
            p = pyaudio.PyAudio()
            
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                
                # Only include devices with input channels
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
            # Fallback: try SpeechRecognition's method
            try:
                mic_list = sr.Microphone.list_microphone_names()
                for i, name in enumerate(mic_list):
                    microphones.append({
                        'index': i,
                        'name': name,
                        'channels': 1,  # Assume mono
                        'sample_rate': 16000,  # Assume standard rate
                        'is_default': i == 0
                    })
            except Exception as e2:
                logging.error(f"Fallback microphone enumeration also failed: {e2}")
        
        return microphones
    
    def __init__(self, whisper_model_size: str = "base", microphone_index: Optional[int] = None):
        """
        Initialize the speech transcriber
        
        Args:
            whisper_model_size: Size of Whisper model ("tiny", "base", "small", "medium", "large")
            microphone_index: Index of microphone device to use (None for default)
        """
        self.whisper_model_size = whisper_model_size
        self.whisper_model = None
        self.microphone_index = microphone_index
        
        # Audio setup
        self.recognizer = sr.Recognizer()
        if microphone_index is not None:
            self.microphone = sr.Microphone(device_index=microphone_index)
        else:
            self.microphone = sr.Microphone()
        
        # Threading and queue management
        self.audio_queue = queue.Queue()
        self.transcription_thread = None
        self.is_listening = False
        
        # Transcription storage
        self.transcription_history: List[TranscriptionSegment] = []
        self.current_transcriptions: Dict[int, str] = {}  # participant_id -> current text
        
        # Participant mapping (face_id -> participant_id)
        self.face_to_participant = {}
        self.next_participant_id = 1
        
        # Audio processing settings
        self.audio_timeout = 1.0  # Timeout for listening
        self.phrase_timeout = 0.3  # Pause between phrases
        self.energy_threshold = 300  # Microphone sensitivity
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self._initialize_audio()
        self._load_whisper_model()
    
    def _initialize_audio(self):
        """Initialize audio settings"""
        try:
            # Test microphone access first
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            self.recognizer.energy_threshold = self.energy_threshold
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = self.phrase_timeout
            
            self.logger.info("Audio initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize audio: {type(e).__name__}: {e}")
    
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
        """Start continuous audio listening"""
        if self.is_listening:
            return
        
        self.is_listening = True
        self.transcription_thread = threading.Thread(target=self._continuous_listen)
        self.transcription_thread.daemon = True
        self.transcription_thread.start()
        
        self.logger.info("Started speech transcription")
    
    def stop_listening(self):
        """Stop audio listening"""
        self.is_listening = False
        
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.transcription_thread.join(timeout=2.0)
        
        self.logger.info("Stopped speech transcription")
    
    def change_microphone(self, microphone_index: Optional[int]) -> bool:
        """
        Change the microphone device
        
        Args:
            microphone_index: Index of new microphone device (None for default)
            
        Returns:
            True if successful, False otherwise
        """
        was_listening = self.is_listening
        
        # Stop listening if active
        if was_listening:
            self.stop_listening()
        
        try:
            # Update microphone
            self.microphone_index = microphone_index
            if microphone_index is not None:
                self.microphone = sr.Microphone(device_index=microphone_index)
            else:
                self.microphone = sr.Microphone()
            
            # Re-initialize audio with new microphone
            self._initialize_audio()
            
            # Restart listening if it was active
            if was_listening:
                self.start_listening()
            
            self.logger.info(f"Changed microphone to device {microphone_index}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to change microphone to device {microphone_index}: {type(e).__name__}: {e}")
            return False
    
    def get_current_microphone_info(self) -> Dict[str, any]:
        """Get information about currently selected microphone"""
        microphones = self.get_available_microphones()
        
        if self.microphone_index is not None:
            for mic in microphones:
                if mic['index'] == self.microphone_index:
                    return mic
        
        # Return default microphone info
        for mic in microphones:
            if mic.get('is_default', False):
                return mic
        
        # Return first available if no default found
        return microphones[0] if microphones else {
            'index': None,
            'name': 'Unknown',
            'channels': 1,
            'sample_rate': 16000,
            'is_default': True
        }
    
    def _continuous_listen(self):
        """Continuous audio listening loop"""
        try:
            while self.is_listening:
                try:
                    # Create a fresh microphone context for each listen attempt
                    with self.microphone as source:
                        # Listen for audio with timeout
                        audio_data = self.recognizer.listen(
                            source, 
                            timeout=self.audio_timeout,
                            phrase_time_limit=5.0  # Max 5 seconds per phrase
                        )
                        
                        # Add to queue for processing
                        self.audio_queue.put({
                            'audio': audio_data,
                            'timestamp': datetime.now()
                        })
                        
                except sr.WaitTimeoutError:
                    # Timeout is normal, continue listening
                    pass
                except Exception as e:
                    self.logger.error(f"Error in continuous listening: {type(e).__name__}: {e}")
                    # Longer sleep on error to prevent rapid error loops
                    time.sleep(0.5)
                        
        except Exception as e:
            self.logger.error(f"Fatal error in continuous listening: {type(e).__name__}: {e}")
    
    def process_audio_queue(self, active_faces: List[int]) -> Dict[int, str]:
        """
        Process queued audio and return new transcriptions
        
        Args:
            active_faces: List of currently detected face IDs
            
        Returns:
            Dictionary mapping participant IDs to new transcription text
        """
        new_transcriptions = {}
        
        # Process all queued audio
        while not self.audio_queue.empty():
            try:
                audio_item = self.audio_queue.get_nowait()
                transcript = self._transcribe_audio(audio_item['audio'])
                
                if transcript and transcript.strip():
                    # Determine which participant is speaking
                    participant_id = self._identify_speaker(active_faces, audio_item['timestamp'])
                    
                    # Store transcription
                    segment = TranscriptionSegment(
                        participant_id=participant_id,
                        text=transcript,
                        timestamp=audio_item['timestamp'],
                        confidence=0.8,  # Placeholder confidence
                        duration=2.0  # Placeholder duration
                    )
                    
                    self.transcription_history.append(segment)
                    
                    # Update current transcriptions
                    if participant_id not in self.current_transcriptions:
                        self.current_transcriptions[participant_id] = ""
                    
                    self.current_transcriptions[participant_id] += f" {transcript}"
                    new_transcriptions[participant_id] = transcript
                    
                    self.logger.info(f"Participant {participant_id}: {transcript}")
                    
            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"Error processing audio: {e}")
        
        return new_transcriptions
    
    def _transcribe_audio(self, audio_data) -> Optional[str]:
        """Transcribe audio using Faster-Whisper"""
        if not self.whisper_model:
            return None
        
        try:
            # Convert audio to numpy array
            audio_np = np.frombuffer(audio_data.get_wav_data(), dtype=np.int16).astype(np.float32) / 32768.0
            
            # Transcribe with Faster-Whisper
            segments, info = self.whisper_model.transcribe(audio_np, language="en")
            
            # Combine all segments
            text = " ".join([segment.text for segment in segments]).strip()
            
            return text if text else None
            
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            return None
    
    def _identify_speaker(self, active_faces: List[int], timestamp: datetime) -> int:
        """
        Identify which participant is speaking based on active faces
        
        Args:
            active_faces: Currently detected face IDs
            timestamp: When the speech was detected
            
        Returns:
            Participant ID
        """
        if not active_faces:
            # No faces detected, use default participant
            return self._get_or_create_participant(0)
        
        if len(active_faces) == 1:
            # Single person, easy mapping
            return self._get_or_create_participant(active_faces[0])
        
        # Multiple faces - for now, use the first one
        # TODO: Implement more sophisticated speaker identification
        return self._get_or_create_participant(active_faces[0])
    
    def _get_or_create_participant(self, face_id: int) -> int:
        """Get or create participant ID for a face ID"""
        if face_id not in self.face_to_participant:
            self.face_to_participant[face_id] = self.next_participant_id
            self.next_participant_id += 1
            self.logger.info(f"Mapped face {face_id} to participant {self.face_to_participant[face_id]}")
        
        return self.face_to_participant[face_id]
    
    def get_participant_transcriptions(self) -> Dict[int, str]:
        """Get current transcriptions for all participants"""
        return self.current_transcriptions.copy()
    
    def get_full_transcript(self) -> str:
        """Get complete transcript of the discussion"""
        transcript_lines = []
        
        for segment in self.transcription_history:
            timestamp_str = segment.timestamp.strftime("%H:%M:%S")
            participant_name = f"Participant {segment.participant_id}"
            transcript_lines.append(f"[{timestamp_str}] {participant_name}: {segment.text}")
        
        return "\n".join(transcript_lines)
    
    def clear_transcriptions(self):
        """Clear all transcription data"""
        self.transcription_history.clear()
        self.current_transcriptions.clear()
        self.face_to_participant.clear()
        self.next_participant_id = 1
        
        self.logger.info("Cleared all transcription data")
    
    def get_discussion_stats(self) -> Dict:
        """Get discussion statistics"""
        if not self.transcription_history:
            return {
                "total_segments": 0,
                "participants": 0,
                "duration_minutes": 0,
                "words_per_participant": {}
            }
        
        # Calculate stats
        participants = set(segment.participant_id for segment in self.transcription_history)
        start_time = min(segment.timestamp for segment in self.transcription_history)
        end_time = max(segment.timestamp for segment in self.transcription_history)
        duration = (end_time - start_time).total_seconds() / 60.0
        
        # Words per participant
        words_per_participant = {}
        for participant_id in participants:
            participant_segments = [s for s in self.transcription_history if s.participant_id == participant_id]
            total_words = sum(len(s.text.split()) for s in participant_segments)
            words_per_participant[f"Participant {participant_id}"] = total_words
        
        return {
            "total_segments": len(self.transcription_history),
            "participants": len(participants),
            "duration_minutes": round(duration, 1),
            "words_per_participant": words_per_participant
        }