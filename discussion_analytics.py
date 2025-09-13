"""
Discussion Analytics Engine
Provides advanced analytics and visualization of discussion patterns
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict, deque
from dataclasses import dataclass

@dataclass
class DiscussionPattern:
    """Represents a discussion pattern analysis"""
    pattern_type: str  # "turn_taking", "interruption_burst", "silence_period", "engagement_peak"
    start_time: float
    end_time: float
    participants: List[int]
    intensity: float
    description: str

class DiscussionAnalytics:
    """Advanced analytics for discussion dynamics"""
    
    def __init__(self):
        self.conversation_flow = deque(maxlen=1000)  # Store conversation flow
        self.engagement_timeline = deque(maxlen=500)  # Store engagement over time
        self.interaction_matrix = defaultdict(lambda: defaultdict(int))  # Who responds to whom
        self.patterns_detected = []
        
    def analyze_discussion_flow(self, activity_tracker: 'ActivityTracker') -> Dict:
        """Analyze overall discussion flow and patterns"""
        
        # Get participant data
        participants = activity_tracker.participants
        activity_history = list(activity_tracker.activity_history)
        
        if not participants or not activity_history:
            return self._empty_analysis()
        
        # Analyze different aspects
        turn_taking_analysis = self._analyze_turn_taking(activity_history)
        engagement_analysis = self._analyze_engagement_patterns(activity_history)
        interaction_analysis = self._analyze_interactions(activity_history)
        balance_analysis = self._analyze_participation_balance(participants)
        
        return {
            'turn_taking': turn_taking_analysis,
            'engagement': engagement_analysis,
            'interactions': interaction_analysis,
            'balance': balance_analysis,
            'patterns': self.patterns_detected[-10:],  # Last 10 patterns
            'conversation_quality': self._calculate_conversation_quality(
                turn_taking_analysis, engagement_analysis, balance_analysis
            )
        }
    
    def _analyze_turn_taking(self, activity_history: List[Dict]) -> Dict:
        """Analyze turn-taking patterns"""
        if len(activity_history) < 10:
            return {'smooth_transitions': 0, 'overlaps': 0, 'interruptions': 0, 'silence_gaps': 0}
        
        smooth_transitions = 0
        overlaps = 0
        interruptions = 0
        silence_gaps = 0
        
        prev_speakers = set()
        
        for i, snapshot in enumerate(activity_history[1:], 1):
            current_speakers = set(snapshot['speaking'])
            prev_snapshot = activity_history[i-1]
            prev_speakers = set(prev_snapshot['speaking'])
            
            # Detect different transition types
            if len(prev_speakers) == 1 and len(current_speakers) == 1 and prev_speakers != current_speakers:
                smooth_transitions += 1
            elif len(current_speakers) > 1:
                overlaps += 1
            elif len(prev_speakers) > 0 and len(current_speakers) > len(prev_speakers):
                interruptions += 1
            elif len(prev_speakers) == 0 and len(current_speakers) == 0:
                silence_gaps += 1
        
        total_transitions = max(1, len(activity_history) - 1)
        
        return {
            'smooth_transitions': smooth_transitions / total_transitions,
            'overlaps': overlaps / total_transitions,
            'interruptions': interruptions / total_transitions,
            'silence_gaps': silence_gaps / total_transitions,
            'transition_quality': smooth_transitions / max(1, smooth_transitions + overlaps + interruptions)
        }
    
    def _analyze_engagement_patterns(self, activity_history: List[Dict]) -> Dict:
        """Analyze engagement patterns over time"""
        if not activity_history:
            return {'peaks': [], 'valleys': [], 'average_engagement': 0, 'engagement_trend': 'stable'}
        
        # Calculate engagement score for each time window
        window_size = 30  # 1 second at 30 FPS
        engagement_scores = []
        timestamps = []
        
        for i in range(0, len(activity_history), window_size):
            window = activity_history[i:i+window_size]
            if not window:
                continue
                
            # Engagement score based on active participants and activity
            avg_participants = np.mean([s['participant_count'] for s in window])
            avg_speakers = np.mean([len(s['speaking']) for s in window])
            
            engagement_score = (avg_speakers / max(1, avg_participants)) * avg_participants
            engagement_scores.append(engagement_score)
            timestamps.append(window[-1]['timestamp'])
        
        if not engagement_scores:
            return {'peaks': [], 'valleys': [], 'average_engagement': 0, 'engagement_trend': 'stable'}
        
        # Find peaks and valleys
        peaks = self._find_peaks(engagement_scores, timestamps)
        valleys = self._find_valleys(engagement_scores, timestamps)
        
        # Calculate trend
        if len(engagement_scores) > 1:
            trend_slope = np.polyfit(range(len(engagement_scores)), engagement_scores, 1)[0]
            if trend_slope > 0.1:
                trend = 'increasing'
            elif trend_slope < -0.1:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'peaks': peaks,
            'valleys': valleys,
            'average_engagement': np.mean(engagement_scores),
            'engagement_trend': trend,
            'engagement_scores': list(zip(timestamps, engagement_scores))
        }
    
    def _analyze_interactions(self, activity_history: List[Dict]) -> Dict:
        """Analyze who responds to whom and interaction patterns"""
        # Track speaker transitions to infer responses
        response_patterns = defaultdict(lambda: defaultdict(int))
        
        prev_speakers = set()
        for snapshot in activity_history:
            current_speakers = set(snapshot['speaking'])
            
            # If someone starts speaking after someone else stops, consider it a response
            if len(prev_speakers) == 1 and len(current_speakers) == 1:
                prev_speaker = list(prev_speakers)[0]
                current_speaker = list(current_speakers)[0]
                if prev_speaker != current_speaker:
                    response_patterns[prev_speaker][current_speaker] += 1
            
            prev_speakers = current_speakers
        
        # Find most common interaction pairs
        interaction_pairs = []
        for speaker1, responses in response_patterns.items():
            for speaker2, count in responses.items():
                if count > 2:  # Only include significant interactions
                    interaction_pairs.append({
                        'from': speaker1,
                        'to': speaker2,
                        'frequency': count
                    })
        
        # Sort by frequency
        interaction_pairs.sort(key=lambda x: x['frequency'], reverse=True)
        
        return {
            'interaction_pairs': interaction_pairs[:10],  # Top 10 interactions
            'response_matrix': dict(response_patterns),
            'most_responsive': self._find_most_responsive(response_patterns),
            'conversation_initiators': self._find_initiators(response_patterns)
        }
    
    def _analyze_participation_balance(self, participants: Dict) -> Dict:
        """Analyze how balanced the participation is"""
        if not participants:
            return {'balance_score': 1.0, 'dominant_speakers': [], 'quiet_participants': []}
        
        speaking_times = [p.total_speaking_time for p in participants.values()]
        
        if not speaking_times or max(speaking_times) == 0:
            return {'balance_score': 1.0, 'dominant_speakers': [], 'quiet_participants': []}
        
        # Calculate balance metrics
        mean_time = np.mean(speaking_times)
        std_time = np.std(speaking_times)
        cv = std_time / mean_time if mean_time > 0 else 0
        
        # Balance score (0-1, higher is better)
        balance_score = max(0, 1 - cv)
        
        # Identify dominant and quiet participants
        threshold_high = mean_time + std_time
        threshold_low = mean_time - std_time
        
        dominant_speakers = []
        quiet_participants = []
        
        for person_id, participant in participants.items():
            if participant.total_speaking_time > threshold_high:
                dominant_speakers.append({
                    'id': person_id,
                    'speaking_time': participant.total_speaking_time,
                    'percentage': (participant.total_speaking_time / sum(speaking_times)) * 100
                })
            elif participant.total_speaking_time < threshold_low:
                quiet_participants.append({
                    'id': person_id,
                    'speaking_time': participant.total_speaking_time,
                    'percentage': (participant.total_speaking_time / sum(speaking_times)) * 100
                })
        
        return {
            'balance_score': balance_score,
            'dominant_speakers': dominant_speakers,
            'quiet_participants': quiet_participants,
            'participation_distribution': [
                {'id': pid, 'time': p.total_speaking_time, 'level': p.participation_level}
                for pid, p in participants.items()
            ]
        }
    
    def _find_peaks(self, scores: List[float], timestamps: List[float]) -> List[Dict]:
        """Find engagement peaks"""
        if len(scores) < 3:
            return []
        
        peaks = []
        for i in range(1, len(scores) - 1):
            if scores[i] > scores[i-1] and scores[i] > scores[i+1] and scores[i] > np.mean(scores):
                peaks.append({
                    'timestamp': timestamps[i],
                    'engagement_score': scores[i],
                    'relative_position': i / len(scores)
                })
        
        return sorted(peaks, key=lambda x: x['engagement_score'], reverse=True)[:5]
    
    def _find_valleys(self, scores: List[float], timestamps: List[float]) -> List[Dict]:
        """Find engagement valleys"""
        if len(scores) < 3:
            return []
        
        valleys = []
        for i in range(1, len(scores) - 1):
            if scores[i] < scores[i-1] and scores[i] < scores[i+1] and scores[i] < np.mean(scores):
                valleys.append({
                    'timestamp': timestamps[i],
                    'engagement_score': scores[i],
                    'relative_position': i / len(scores)
                })
        
        return sorted(valleys, key=lambda x: x['engagement_score'])[:5]
    
    def _find_most_responsive(self, response_patterns: Dict) -> List[Dict]:
        """Find participants who respond most to others"""
        responsiveness = defaultdict(int)
        
        for speaker1, responses in response_patterns.items():
            for speaker2, count in responses.items():
                responsiveness[speaker2] += count
        
        sorted_responsive = sorted(responsiveness.items(), key=lambda x: x[1], reverse=True)
        return [{'id': pid, 'response_count': count} for pid, count in sorted_responsive[:5]]
    
    def _find_initiators(self, response_patterns: Dict) -> List[Dict]:
        """Find participants who initiate conversations most"""
        initiation_count = defaultdict(int)
        
        for speaker1, responses in response_patterns.items():
            total_responses = sum(responses.values())
            initiation_count[speaker1] = total_responses
        
        sorted_initiators = sorted(initiation_count.items(), key=lambda x: x[1], reverse=True)
        return [{'id': pid, 'initiation_count': count} for pid, count in sorted_initiators[:5]]
    
    def _calculate_conversation_quality(self, turn_taking: Dict, engagement: Dict, balance: Dict) -> Dict:
        """Calculate overall conversation quality metrics"""
        
        # Normalize scores to 0-1 range
        turn_quality = turn_taking.get('transition_quality', 0)
        engagement_quality = min(1.0, engagement.get('average_engagement', 0) / 2.0)  # Normalize assuming max 2
        balance_quality = balance.get('balance_score', 0)
        
        # Weighted average
        overall_quality = (turn_quality * 0.4 + engagement_quality * 0.3 + balance_quality * 0.3)
        
        # Determine quality category
        if overall_quality >= 0.8:
            quality_category = "Excellent"
        elif overall_quality >= 0.6:
            quality_category = "Good"
        elif overall_quality >= 0.4:
            quality_category = "Fair"
        else:
            quality_category = "Needs Improvement"
        
        return {
            'overall_score': overall_quality,
            'category': quality_category,
            'turn_taking_score': turn_quality,
            'engagement_score': engagement_quality,
            'balance_score': balance_quality,
            'recommendations': self._generate_recommendations(turn_taking, engagement, balance)
        }
    
    def _generate_recommendations(self, turn_taking: Dict, engagement: Dict, balance: Dict) -> List[str]:
        """Generate recommendations for improving discussion quality"""
        recommendations = []
        
        # Turn-taking recommendations
        if turn_taking.get('overlaps', 0) > 0.3:
            recommendations.append("Consider using techniques to reduce interruptions and overlapping speech")
        
        if turn_taking.get('silence_gaps', 0) > 0.2:
            recommendations.append("Try to maintain more consistent conversation flow with fewer silent gaps")
        
        # Engagement recommendations
        if engagement.get('average_engagement', 0) < 1.0:
            recommendations.append("Encourage more participants to actively engage in the discussion")
        
        # Balance recommendations
        if balance.get('balance_score', 0) < 0.6:
            recommendations.append("Work on balancing participation - encourage quiet members and moderate dominant speakers")
        
        if len(balance.get('dominant_speakers', [])) > 0:
            recommendations.append("Some participants are dominating the conversation - consider time management")
        
        if len(balance.get('quiet_participants', [])) > 1:
            recommendations.append("Several participants are quiet - actively encourage their participation")
        
        return recommendations
    
    def _empty_analysis(self) -> Dict:
        """Return empty analysis structure"""
        return {
            'turn_taking': {'smooth_transitions': 0, 'overlaps': 0, 'interruptions': 0, 'silence_gaps': 0},
            'engagement': {'peaks': [], 'valleys': [], 'average_engagement': 0, 'engagement_trend': 'stable'},
            'interactions': {'interaction_pairs': [], 'response_matrix': {}, 'most_responsive': [], 'conversation_initiators': []},
            'balance': {'balance_score': 1.0, 'dominant_speakers': [], 'quiet_participants': []},
            'patterns': [],
            'conversation_quality': {'overall_score': 0, 'category': 'No Data', 'recommendations': []}
        }
    
    def generate_discussion_map(self, activity_tracker: 'ActivityTracker') -> Dict:
        """Generate a visual map of discussion activity"""
        participants = activity_tracker.participants
        activity_history = list(activity_tracker.activity_history)
        
        if not participants or not activity_history:
            return {'timeline': [], 'heatmap_data': [], 'flow_diagram': []}
        
        # Create timeline data
        timeline_data = []
        for snapshot in activity_history[-100:]:  # Last 100 snapshots
            timeline_data.append({
                'timestamp': snapshot['timestamp'],
                'active_speakers': snapshot['speaking'],
                'participant_count': snapshot['participant_count']
            })
        
        # Create heatmap data (participation intensity over time)
        heatmap_data = self._create_participation_heatmap(timeline_data, participants)
        
        # Create flow diagram data
        flow_data = self._create_flow_diagram_data(activity_history)
        
        return {
            'timeline': timeline_data,
            'heatmap_data': heatmap_data,
            'flow_diagram': flow_data,
            'statistics': {
                'total_duration': time.time() - activity_tracker.discussion_start_time,
                'active_periods': len([s for s in activity_history if s['speaking']]),
                'silent_periods': len([s for s in activity_history if not s['speaking']])
            }
        }
    
    def _create_participation_heatmap(self, timeline_data: List[Dict], participants: Dict) -> List[Dict]:
        """Create heatmap data for participation visualization"""
        # Group timeline into time windows
        window_size = 10  # seconds
        heatmap_data = []
        
        if not timeline_data:
            return heatmap_data
        
        start_time = timeline_data[0]['timestamp']
        current_window_start = start_time
        
        for participant_id in participants.keys():
            activity_windows = []
            
            for snapshot in timeline_data:
                window_index = int((snapshot['timestamp'] - start_time) // window_size)
                
                # Ensure we have enough windows
                while len(activity_windows) <= window_index:
                    activity_windows.append(0)
                
                # Mark activity in this window
                if participant_id in snapshot['active_speakers']:
                    activity_windows[window_index] += 1
            
            heatmap_data.append({
                'participant_id': participant_id,
                'activity_windows': activity_windows,
                'total_activity': sum(activity_windows)
            })
        
        return heatmap_data
    
    def _create_flow_diagram_data(self, activity_history: List[Dict]) -> List[Dict]:
        """Create flow diagram showing conversation transitions"""
        transitions = []
        
        prev_speakers = set()
        for snapshot in activity_history:
            current_speakers = set(snapshot['speaking'])
            
            if prev_speakers != current_speakers:
                transitions.append({
                    'timestamp': snapshot['timestamp'],
                    'from_speakers': list(prev_speakers),
                    'to_speakers': list(current_speakers),
                    'transition_type': self._classify_transition(prev_speakers, current_speakers)
                })
            
            prev_speakers = current_speakers
        
        return transitions[-50:]  # Last 50 transitions
    
    def _classify_transition(self, prev_speakers: set, current_speakers: set) -> str:
        """Classify the type of speaking transition"""
        if not prev_speakers and current_speakers:
            return "conversation_start"
        elif prev_speakers and not current_speakers:
            return "conversation_pause"
        elif len(prev_speakers) == 1 and len(current_speakers) == 1 and prev_speakers != current_speakers:
            return "turn_taking"
        elif len(current_speakers) > len(prev_speakers):
            return "interruption"
        elif len(current_speakers) < len(prev_speakers):
            return "speaker_dropout"
        else:
            return "continuation"