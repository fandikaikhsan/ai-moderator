"""
Speaking Activity Tracker Module
Tracks and analyzes speaking patterns, identifies dominant and quiet participants
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
from datetime import datetime

@dataclass
class ParticipantStats:
    """Statistics for a single participant"""
    person_id: int
    total_speaking_time: float = 0.0
    speaking_sessions: List[Tuple[float, float]] = field(default_factory=list)  # (start_time, end_time)
    current_session_start: Optional[float] = None
    interruptions_made: int = 0
    times_interrupted: int = 0
    average_speaking_intensity: float = 0.0
    max_speaking_intensity: float = 0.0
    speaking_turns: int = 0
    last_activity_time: float = 0.0
    participation_level: str = "unknown"  # "dominant", "balanced", "quiet", "silent"

@dataclass
class DiscussionMetrics:
    """Overall discussion metrics"""
    total_participants: int = 0
    discussion_duration: float = 0.0
    total_speaking_time: float = 0.0
    silence_time: float = 0.0
    overlapping_speech_time: float = 0.0
    turn_taking_balance: float = 0.0  # 0-1, higher is more balanced
    most_active_participant: Optional[int] = None
    quietest_participant: Optional[int] = None
    discussion_quality_score: float = 0.0  # 0-1, based on participation balance

class ActivityTracker:
    """Tracks speaking activity and analyzes discussion dynamics"""
    
    def __init__(self, session_gap_threshold: float = 2.0):
        """
        Initialize activity tracker
        
        Args:
            session_gap_threshold: Seconds of silence to consider end of speaking session
        """
        self.session_gap_threshold = session_gap_threshold
        self.participants: Dict[int, ParticipantStats] = {}
        self.discussion_start_time = time.time()
        self.last_update_time = time.time()
        
        # Real-time tracking
        self.activity_history = deque(maxlen=300)  # 10 seconds at 30 FPS
        self.intensity_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Interruption detection
        self.speaking_state_history = deque(maxlen=30)  # 1 second at 30 FPS
        
        # Transcript tracking - detailed timeline of speaking events
        self.transcript_log = []  # List of speaking events with timestamps
        self.last_speaking_state = set()  # Track previous speaking state for change detection
        
    def update_activity(self, speech_activities: Dict[int, 'SpeechActivity']):
        """
        Update activity tracking with current speech activities
        
        Args:
            speech_activities: Dictionary mapping person ID to SpeechActivity
        """
        current_time = time.time()
        
        # Initialize participants if needed
        for person_id in speech_activities.keys():
            if person_id not in self.participants:
                self.participants[person_id] = ParticipantStats(person_id=person_id)
        
        # Track current speaking state
        currently_speaking = []
        for person_id, activity in speech_activities.items():
            if activity.is_speaking:
                currently_speaking.append(person_id)
                self._update_speaking_session(person_id, current_time, activity)
            else:
                self._end_speaking_session(person_id, current_time)
            
            # Update intensity history
            self.intensity_history[person_id].append(activity.speaking_intensity)
            
            # Update participant stats
            participant = self.participants[person_id]
            participant.last_activity_time = current_time
            if activity.speaking_intensity > participant.max_speaking_intensity:
                participant.max_speaking_intensity = activity.speaking_intensity
        
        # Detect interruptions
        self._detect_interruptions(currently_speaking, current_time)
        
        # Store activity snapshot
        activity_snapshot = {
            'timestamp': current_time,
            'speaking': currently_speaking,
            'participant_count': len(speech_activities),
            'overlapping_speakers': len(currently_speaking)
        }
        self.activity_history.append(activity_snapshot)
        
        # Log transcript events (speaking state changes)
        self._log_transcript_events(currently_speaking, current_time)
        
        # Update participation levels
        self._update_participation_levels()
        
        self.last_update_time = current_time
    
    def _update_speaking_session(self, person_id: int, current_time: float, activity):
        """Update ongoing speaking session for a participant"""
        participant = self.participants[person_id]
        
        # Start new session if needed
        if participant.current_session_start is None:
            participant.current_session_start = current_time
            participant.speaking_turns += 1
        
        # Update total speaking time (approximate)
        time_delta = current_time - self.last_update_time
        participant.total_speaking_time += time_delta
        
        # Update average intensity
        if len(self.intensity_history[person_id]) > 0:
            participant.average_speaking_intensity = np.mean(list(self.intensity_history[person_id]))
    
    def _end_speaking_session(self, person_id: int, current_time: float):
        """End speaking session for a participant"""
        participant = self.participants[person_id]
        
        if participant.current_session_start is not None:
            # Check if gap is long enough to end session
            gap_duration = current_time - participant.current_session_start
            if gap_duration > self.session_gap_threshold:
                # End session
                session_duration = current_time - participant.current_session_start
                participant.speaking_sessions.append(
                    (participant.current_session_start, current_time)
                )
                participant.current_session_start = None
    
    def _log_transcript_events(self, currently_speaking: List[int], current_time: float):
        """Log transcript events when speaking state changes"""
        current_speakers_set = set(currently_speaking)
        
        # Check for speaking state changes
        if current_speakers_set != self.last_speaking_state:
            # Calculate relative time from discussion start
            relative_time = current_time - self.discussion_start_time
            
            # Log speaking started events
            started_speaking = current_speakers_set - self.last_speaking_state
            for person_id in started_speaking:
                self.transcript_log.append({
                    'timestamp': current_time,
                    'relative_time': relative_time,
                    'event_type': 'speaking_started',
                    'person_id': person_id,
                    'person_name': f"Person {person_id}",
                    'concurrent_speakers': list(current_speakers_set),
                    'is_interruption': len(self.last_speaking_state) > 0
                })
            
            # Log speaking stopped events
            stopped_speaking = self.last_speaking_state - current_speakers_set
            for person_id in stopped_speaking:
                self.transcript_log.append({
                    'timestamp': current_time,
                    'relative_time': relative_time,
                    'event_type': 'speaking_stopped',
                    'person_id': person_id,
                    'person_name': f"Person {person_id}",
                    'concurrent_speakers': list(current_speakers_set),
                    'was_interrupted': person_id in self.last_speaking_state and len(current_speakers_set) > 0
                })
            
            # Update last speaking state
            self.last_speaking_state = current_speakers_set.copy()
    
    def _detect_interruptions(self, current_speakers: List[int], current_time: float):
        """Detect and track interruptions"""
        current_speakers_set = set(current_speakers)
        
        # Get previous speaking state
        if len(self.speaking_state_history) > 0:
            prev_speakers = set(self.speaking_state_history[-1])
        else:
            prev_speakers = set()
        
        # Store current state
        self.speaking_state_history.append(current_speakers)
        
        # Detect new speakers joining ongoing conversation
        new_speakers = current_speakers_set - prev_speakers
        
        if len(prev_speakers) > 0 and len(new_speakers) > 0:
            # Someone started speaking while others were already talking
            for interrupter in new_speakers:
                if interrupter in self.participants:
                    self.participants[interrupter].interruptions_made += 1
            
            # Mark interrupted participants
            for interrupted in prev_speakers:
                if interrupted in self.participants and interrupted not in current_speakers:
                    self.participants[interrupted].times_interrupted += 1
    
    def _update_participation_levels(self):
        """Update participation level categories for all participants"""
        if not self.participants:
            return
        
        # Calculate discussion duration
        discussion_duration = time.time() - self.discussion_start_time
        
        # Get speaking time percentages
        speaking_percentages = {}
        total_speaking_time = sum(p.total_speaking_time for p in self.participants.values())
        
        for person_id, participant in self.participants.items():
            if discussion_duration > 0:
                speaking_percentages[person_id] = participant.total_speaking_time / discussion_duration
            else:
                speaking_percentages[person_id] = 0
        
        # Categorize participants
        if speaking_percentages:
            avg_percentage = np.mean(list(speaking_percentages.values()))
            std_percentage = np.std(list(speaking_percentages.values()))
            
            for person_id, percentage in speaking_percentages.items():
                participant = self.participants[person_id]
                
                if percentage > avg_percentage + std_percentage:
                    participant.participation_level = "dominant"
                elif percentage > avg_percentage - std_percentage:
                    participant.participation_level = "balanced"
                elif percentage > 0.01:  # At least 1% speaking time
                    participant.participation_level = "quiet"
                else:
                    participant.participation_level = "silent"
    
    def get_discussion_metrics(self) -> DiscussionMetrics:
        """Generate comprehensive discussion metrics"""
        current_time = time.time()
        discussion_duration = current_time - self.discussion_start_time
        
        # Calculate total speaking time and silence
        total_speaking_time = sum(p.total_speaking_time for p in self.participants.values())
        silence_time = max(0, discussion_duration - total_speaking_time)
        
        # Calculate overlapping speech time
        overlapping_time = 0
        for snapshot in self.activity_history:
            if snapshot['overlapping_speakers'] > 1:
                overlapping_time += 1/30.0  # Assume 30 FPS
        
        # Find most active and quietest participants
        most_active = None
        quietest = None
        max_speaking_time = 0
        min_speaking_time = float('inf')
        
        for person_id, participant in self.participants.items():
            if participant.total_speaking_time > max_speaking_time:
                max_speaking_time = participant.total_speaking_time
                most_active = person_id
            if participant.total_speaking_time < min_speaking_time:
                min_speaking_time = participant.total_speaking_time
                quietest = person_id
        
        # Calculate turn-taking balance
        turn_taking_balance = self._calculate_turn_taking_balance()
        
        # Calculate discussion quality score
        quality_score = self._calculate_discussion_quality()
        
        return DiscussionMetrics(
            total_participants=len(self.participants),
            discussion_duration=discussion_duration,
            total_speaking_time=total_speaking_time,
            silence_time=silence_time,
            overlapping_speech_time=overlapping_time,
            turn_taking_balance=turn_taking_balance,
            most_active_participant=most_active,
            quietest_participant=quietest,
            discussion_quality_score=quality_score
        )
    
    def _calculate_turn_taking_balance(self) -> float:
        """Calculate how balanced the turn-taking is (0-1, higher is better)"""
        if len(self.participants) < 2:
            return 1.0
        
        speaking_times = [p.total_speaking_time for p in self.participants.values()]
        if max(speaking_times) == 0:
            return 1.0
        
        # Calculate coefficient of variation (lower is more balanced)
        mean_time = np.mean(speaking_times)
        std_time = np.std(speaking_times)
        
        if mean_time == 0:
            return 1.0
        
        cv = std_time / mean_time
        # Convert to 0-1 scale where 1 is perfectly balanced
        balance_score = max(0, 1 - cv)
        return balance_score
    
    def _calculate_discussion_quality(self) -> float:
        """Calculate overall discussion quality score (0-1)"""
        if not self.participants:
            return 0.0
        
        # Calculate balance score directly to avoid circular dependency
        balance_score = self._calculate_turn_taking_balance()
        
        # Calculate other values directly
        total_speaking_time = sum(p.total_speaking_time for p in self.participants.values())
        discussion_duration = time.time() - self.discussion_start_time
        overlapping_speech_time = 0.0  # Simplified for now - could calculate from transcript log
        
        # Participation rate (how much of total time was spent speaking)
        participation_rate = min(1.0, total_speaking_time / max(1, discussion_duration))
        
        # Interruption penalty
        total_interruptions = sum(p.interruptions_made for p in self.participants.values())
        interruption_penalty = max(0, 1 - (total_interruptions / max(1, len(self.participants) * 10)))
        
        # Overlap penalty (some overlap is natural, too much is chaotic)
        overlap_ratio = overlapping_speech_time / max(1, total_speaking_time)
        overlap_penalty = max(0, 1 - max(0, overlap_ratio - 0.1) * 2)  # Penalty starts at 10% overlap
        
        # Combine factors
        quality_score = (balance_score * 0.4 + 
                        participation_rate * 0.3 + 
                        interruption_penalty * 0.2 + 
                        overlap_penalty * 0.1)
        
        return min(1.0, quality_score)
    
    def get_participant_rankings(self) -> List[Tuple[int, ParticipantStats]]:
        """Get participants ranked by speaking time"""
        return sorted(self.participants.items(), 
                     key=lambda x: x[1].total_speaking_time, 
                     reverse=True)
    
    def get_activity_matrix(self) -> Dict[int, Dict[str, float]]:
        """Get activity matrix for UI display"""
        matrix = {}
        current_time = time.time()
        discussion_duration = max(1, current_time - self.discussion_start_time)
        
        for person_id, participant in self.participants.items():
            speaking_percentage = (participant.total_speaking_time / discussion_duration) * 100
            
            matrix[person_id] = {
                'speaking_percentage': speaking_percentage,
                'speaking_time': participant.total_speaking_time,
                'speaking_turns': participant.speaking_turns,
                'interruptions_made': participant.interruptions_made,
                'times_interrupted': participant.times_interrupted,
                'average_intensity': participant.average_speaking_intensity,
                'participation_level': participant.participation_level,
                'is_active_now': (current_time - participant.last_activity_time) < 1.0
            }
        
        return matrix
    
    def export_session_data(self) -> Dict:
        """Export session data for analysis or reporting"""
        metrics = self.get_discussion_metrics()
        activity_matrix = self.get_activity_matrix()
        
        session_data = {
            'session_info': {
                'start_time': self.discussion_start_time,
                'duration': metrics.discussion_duration,
                'export_time': time.time()
            },
            'metrics': {
                'total_participants': metrics.total_participants,
                'total_speaking_time': metrics.total_speaking_time,
                'silence_time': metrics.silence_time,
                'overlapping_speech_time': metrics.overlapping_speech_time,
                'turn_taking_balance': metrics.turn_taking_balance,
                'discussion_quality_score': metrics.discussion_quality_score
            },
            'participants': {},
            'activity_matrix': activity_matrix,
            'transcript': {
                'events': self.transcript_log,
                'total_events': len(self.transcript_log),
                'summary': self._generate_transcript_summary()
            }
        }
        
        # Export detailed participant data
        for person_id, participant in self.participants.items():
            session_data['participants'][person_id] = {
                'total_speaking_time': participant.total_speaking_time,
                'speaking_sessions': participant.speaking_sessions,
                'speaking_turns': participant.speaking_turns,
                'interruptions_made': participant.interruptions_made,
                'times_interrupted': participant.times_interrupted,
                'average_speaking_intensity': participant.average_speaking_intensity,
                'max_speaking_intensity': participant.max_speaking_intensity,
                'participation_level': participant.participation_level
            }
        
        return session_data
    
    def _generate_transcript_summary(self) -> Dict:
        """Generate a summary of the transcript events"""
        if not self.transcript_log:
            return {
                'total_speaking_events': 0,
                'interruptions': 0,
                'speaker_changes': 0,
                'longest_speaking_session': 0.0
            }
        
        speaking_starts = [event for event in self.transcript_log if event['event_type'] == 'speaking_started']
        speaking_stops = [event for event in self.transcript_log if event['event_type'] == 'speaking_stopped']
        interruptions = [event for event in speaking_starts if event['is_interruption']]
        
        # Calculate speaking session durations
        session_durations = []
        for start_event in speaking_starts:
            person_id = start_event['person_id']
            start_time = start_event['relative_time']
            
            # Find corresponding stop event
            for stop_event in speaking_stops:
                if (stop_event['person_id'] == person_id and 
                    stop_event['relative_time'] > start_time):
                    duration = stop_event['relative_time'] - start_time
                    session_durations.append(duration)
                    break
        
        return {
            'total_speaking_events': len(speaking_starts),
            'interruptions': len(interruptions),
            'speaker_changes': len(speaking_starts) + len(speaking_stops),
            'longest_speaking_session': max(session_durations) if session_durations else 0.0,
            'average_speaking_session': sum(session_durations) / len(session_durations) if session_durations else 0.0
        }
    
    def reset_session(self):
        """Reset all tracking data for a new session"""
        self.participants.clear()
        self.activity_history.clear()
        self.intensity_history.clear()
        self.speaking_state_history.clear()
        self.transcript_log.clear()
        self.last_speaking_state.clear()
        self.discussion_start_time = time.time()
        self.last_update_time = time.time()