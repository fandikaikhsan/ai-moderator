#!/usr/bin/env python3
"""
Lightweight Discussion Summarizer
Fast, local summarization without heavy LLM dependencies
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from collections import Counter, defaultdict

@dataclass
class DiscussionSummary:
    """Represents a discussion summary"""
    summary: str
    key_points: List[str]
    participants_contribution: Dict[str, str]
    action_items: List[str]
    sentiment: str
    duration_minutes: float
    generated_at: datetime

class LightweightSummarizer:
    """Fast, lightweight discussion summarizer using rule-based analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Keywords for different categories
        self.action_keywords = [
            'will', 'should', 'need to', 'must', 'have to', 'going to',
            'plan to', 'decide', 'assign', 'schedule', 'deadline',
            'action', 'task', 'todo', 'follow up', 'next step'
        ]
        
        self.question_keywords = [
            'what', 'how', 'when', 'where', 'why', 'which', 'who',
            'can we', 'should we', 'do you think', 'any thoughts'
        ]
        
        self.positive_keywords = [
            'great', 'good', 'excellent', 'perfect', 'agree', 'yes',
            'fantastic', 'awesome', 'wonderful', 'love', 'like'
        ]
        
        self.negative_keywords = [
            'no', 'problem', 'issue', 'concern', 'worry', 'difficult',
            'challenge', 'disagree', 'wrong', 'bad', 'terrible'
        ]
        
        self.topic_keywords = {
            'timeline': ['timeline', 'schedule', 'deadline', 'date', 'time', 'when'],
            'budget': ['budget', 'cost', 'money', 'price', 'expense', 'funding'],
            'team': ['team', 'member', 'people', 'person', 'staff', 'role'],
            'project': ['project', 'deliverable', 'milestone', 'goal', 'objective'],
            'meeting': ['meeting', 'discuss', 'agenda', 'presentation', 'review'],
            'decision': ['decide', 'choice', 'option', 'alternative', 'solution']
        }
    
    def generate_summary(self, transcript: str, discussion_stats: Dict) -> DiscussionSummary:
        """
        Generate a fast summary using rule-based analysis
        
        Args:
            transcript: Full transcript of the discussion
            discussion_stats: Statistics about the discussion
            
        Returns:
            DiscussionSummary object with analysis results
        """
        if not transcript.strip():
            return self._create_empty_summary()
        
        # Parse transcript
        segments = self._parse_transcript(transcript)
        
        if not segments:
            return self._create_empty_summary()
        
        # Analyze participants
        participants = self._analyze_participants(segments)
        
        # Extract key topics
        key_topics = self._extract_key_topics(segments)
        
        # Find action items
        action_items = self._extract_action_items(segments)
        
        # Analyze sentiment
        sentiment = self._analyze_sentiment(segments)
        
        # Generate summary text
        summary_text = self._generate_summary_text(segments, participants, key_topics)
        
        # Create key points
        key_points = self._create_key_points(segments, participants, key_topics)
        
        # Get duration
        duration = discussion_stats.get('duration_minutes', self._estimate_duration(segments))
        
        return DiscussionSummary(
            summary=summary_text,
            key_points=key_points,
            participants_contribution=participants,
            action_items=action_items,
            sentiment=sentiment,
            duration_minutes=duration,
            generated_at=datetime.now()
        )
    
    def _parse_transcript(self, transcript: str) -> List[Dict]:
        """Parse transcript into segments"""
        segments = []
        # Split by actual newlines, not literal \\n
        lines = transcript.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to parse format: [timestamp] Participant X: text
            match = re.match(r'\[(.*?)\]\s*(.*?):\s*(.*)', line)
            if match:
                timestamp_str, participant, text = match.groups()
                segments.append({
                    'timestamp': timestamp_str,
                    'participant': participant.strip(),
                    'text': text.strip().lower(),
                    'original_text': text.strip()
                })
                continue
            
            # Try alternative format: Participant X: text (no timestamp)  
            if ':' in line and not line.startswith('['):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    participant = parts[0].strip()
                    text = parts[1].strip()
                    if participant and text:  # Make sure both exist
                        segments.append({
                            'timestamp': '',
                            'participant': participant,
                            'text': text.lower(),
                            'original_text': text
                        })
                        continue
        
        return segments
    
    def _analyze_participants(self, segments: List[Dict]) -> Dict[str, str]:
        """Analyze participant contributions"""
        participant_stats = defaultdict(list)
        
        for segment in segments:
            participant = segment['participant']
            text = segment['text']
            participant_stats[participant].append(text)
        
        contributions = {}
        for participant, texts in participant_stats.items():
            total_words = sum(len(text.split()) for text in texts)
            contribution_count = len(texts)
            
            # Categorize contribution level
            if contribution_count >= 5:
                level = "Very Active"
            elif contribution_count >= 3:
                level = "Active"
            elif contribution_count >= 1:
                level = "Moderate"
            else:
                level = "Quiet"
            
            contributions[participant] = f"{level} ({contribution_count} contributions, ~{total_words} words)"
        
        return contributions
    
    def _extract_key_topics(self, segments: List[Dict]) -> List[str]:
        """Extract key topics from discussion"""
        topic_scores = defaultdict(int)
        
        for segment in segments:
            text = segment['text']
            words = text.split()
            
            # Score topics based on keyword matches
            for topic, keywords in self.topic_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        topic_scores[topic] += 1
        
        # Get top topics
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, score in sorted_topics[:3] if score > 0]
    
    def _extract_action_items(self, segments: List[Dict]) -> List[str]:
        """Extract potential action items"""
        action_items = []
        
        for segment in segments:
            text = segment['text']
            original_text = segment['original_text']
            
            # Look for action keywords
            for keyword in self.action_keywords:
                if keyword in text:
                    # Extract the sentence containing the action
                    sentences = original_text.split('.')
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            clean_sentence = sentence.strip()
                            if len(clean_sentence) > 10 and clean_sentence not in action_items:
                                action_items.append(clean_sentence)
                            break
                    break
        
        return action_items[:5]  # Limit to 5 action items
    
    def _analyze_sentiment(self, segments: List[Dict]) -> str:
        """Analyze overall sentiment"""
        positive_count = 0
        negative_count = 0
        
        for segment in segments:
            text = segment['text']
            
            for keyword in self.positive_keywords:
                if keyword in text:
                    positive_count += 1
            
            for keyword in self.negative_keywords:
                if keyword in text:
                    negative_count += 1
        
        if positive_count > negative_count * 1.5:
            return "Positive"
        elif negative_count > positive_count * 1.5:
            return "Negative"
        else:
            return "Neutral"
    
    def _generate_summary_text(self, segments: List[Dict], participants: Dict, topics: List[str]) -> str:
        """Generate summary text"""
        participant_count = len(participants)
        total_contributions = len(segments)
        
        summary = f"Discussion involved {participant_count} participant{'s' if participant_count != 1 else ''} with {total_contributions} total contributions. "
        
        if topics:
            summary += f"Main topics discussed: {', '.join(topics)}. "
        
        # Add participation insight
        if participant_count > 1:
            active_participants = [p for p, desc in participants.items() if 'Very Active' in desc or 'Active' in desc]
            if active_participants:
                summary += f"Most active participants: {', '.join(active_participants)}."
        
        return summary
    
    def _create_key_points(self, segments: List[Dict], participants: Dict, topics: List[str]) -> List[str]:
        """Create key points list"""
        key_points = []
        
        # Participation summary
        if len(participants) > 1:
            key_points.append(f"{len(participants)} participants engaged in discussion")
        
        # Topic coverage
        if topics:
            key_points.append(f"Topics covered: {', '.join(topics)}")
        
        # Contribution distribution
        active_count = sum(1 for desc in participants.values() if 'Very Active' in desc or 'Active' in desc)
        if active_count > 0:
            key_points.append(f"{active_count} participants were actively engaged")
        
        # Content analysis
        total_words = sum(len(segment['text'].split()) for segment in segments)
        key_points.append(f"Discussion contained approximately {total_words} words")
        
        return key_points
    
    def _estimate_duration(self, segments: List[Dict]) -> float:
        """Estimate discussion duration based on content"""
        total_words = sum(len(segment['text'].split()) for segment in segments)
        # Assume average speaking rate of 150 words per minute
        estimated_minutes = total_words / 150
        return max(1, round(estimated_minutes, 1))
    
    def _create_empty_summary(self) -> DiscussionSummary:
        """Create empty summary when no content available"""
        return DiscussionSummary(
            summary="No discussion content available to summarize.",
            key_points=["No discussion recorded"],
            participants_contribution={"Info": "No participants detected"},
            action_items=["No action items identified"],
            sentiment="Neutral",
            duration_minutes=0,
            generated_at=datetime.now()
        )