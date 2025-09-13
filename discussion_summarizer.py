"""
Discussion Summarizer Module for AI Discussion Moderator
Uses Ollama with local Llama models for discussion analysis and summarization
"""

import ollama
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

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

class DiscussionSummarizer:
    """Handles discussion analysis and summarization using local Ollama/Llama"""
    
    def __init__(self, model_name: str = "llama3.2"):
        """
        Initialize the discussion summarizer
        
        Args:
            model_name: Name of the Ollama model to use (e.g., "llama3.2", "llama2", "mistral")
        """
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        # Check if Ollama is available
        self._check_ollama_availability()
    
    def _check_ollama_availability(self) -> bool:
        """Check if Ollama service is running and model is available"""
        try:
            # Try to list available models
            models = ollama.list()
            available_models = [model['name'] for model in models.get('models', [])]
            
            if not available_models:
                self.logger.warning("No models found in Ollama")
                self.logger.info(f"To install a model, run: ollama pull {self.model_name}")
                return False
            
            # Check if our preferred model exists, or use any available model
            if self.model_name not in available_models:
                # Try to find any compatible model
                compatible_models = [m for m in available_models if any(x in m for x in ['llama', 'mistral', 'phi'])]
                if compatible_models:
                    self.model_name = compatible_models[0]
                    self.logger.info(f"Using available model: {self.model_name}")
                else:
                    self.logger.warning(f"No compatible models found. Available: {available_models}")
                    return False
            
            self.logger.info(f"Ollama service ready with model '{self.model_name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Ollama service not available: {e}")
            self.logger.info("Please ensure Ollama is installed and running: https://ollama.ai/")
            return False
    
    def generate_summary(self, transcript: str, discussion_stats: Dict) -> DiscussionSummary:
        """
        Generate a comprehensive discussion summary
        
        Args:
            transcript: Full transcript of the discussion
            discussion_stats: Statistics about the discussion
            
        Returns:
            DiscussionSummary object with analysis results
        """
        if not transcript.strip():
            return self._create_empty_summary()
        
        # Check if Ollama is available
        if not self._check_ollama_availability():
            return self._create_fallback_summary(transcript, discussion_stats)
        
        try:
            # Generate main summary
            summary = self._generate_main_summary(transcript)
            
            # Extract key points
            key_points = self._extract_key_points(transcript)
            
            # Analyze participant contributions
            participants_contribution = self._analyze_participants(transcript, discussion_stats)
            
            # Extract action items
            action_items = self._extract_action_items(transcript)
            
            # Analyze sentiment
            sentiment = self._analyze_sentiment(transcript)
            
            return DiscussionSummary(
                summary=summary,
                key_points=key_points,
                participants_contribution=participants_contribution,
                action_items=action_items,
                sentiment=sentiment,
                duration_minutes=discussion_stats.get('duration_minutes', 0),
                generated_at=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return self._create_fallback_summary(transcript, discussion_stats)
    
    def _generate_main_summary(self, transcript: str) -> str:
        """Generate main discussion summary"""
        prompt = f"""
Please provide a concise summary of the following discussion transcript. 
Focus on the main topics discussed, key decisions made, and overall flow of the conversation.
Keep the summary to 2-3 paragraphs.

Transcript:
{transcript}

Summary:"""
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            
            return response['message']['content'].strip()
            
        except Exception as e:
            self.logger.error(f"Error generating main summary: {e}")
            return "Unable to generate summary due to technical issues."
    
    def _extract_key_points(self, transcript: str) -> List[str]:
        """Extract key points from the discussion"""
        prompt = f"""
Analyze the following discussion transcript and extract 5-7 key points or main topics that were discussed.
Return only the key points as a simple list, one point per line.

Transcript:
{transcript}

Key Points:"""
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            
            content = response['message']['content'].strip()
            # Parse the response into a list
            key_points = [point.strip('- ').strip() for point in content.split('\n') if point.strip()]
            return key_points[:7]  # Limit to 7 points
            
        except Exception as e:
            self.logger.error(f"Error extracting key points: {e}")
            return ["Unable to extract key points"]
    
    def _analyze_participants(self, transcript: str, discussion_stats: Dict) -> Dict[str, str]:
        """Analyze each participant's contribution"""
        participants_data = discussion_stats.get('words_per_participant', {})
        
        if not participants_data:
            return {"Overall": "Participant analysis not available"}
        
        prompt = f"""
Analyze the following discussion transcript and provide a brief analysis of each participant's contribution.
Focus on their main topics, communication style, and level of engagement.
Keep each analysis to 1-2 sentences.

Participants word counts: {participants_data}

Transcript:
{transcript}

Participant Analysis:"""
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            
            content = response['message']['content'].strip()
            
            # Parse the response - try to extract participant-specific analysis
            analysis = {}
            lines = content.split('\n')
            
            for line in lines:
                if 'Participant' in line and ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        participant = parts[0].strip()
                        contribution = parts[1].strip()
                        analysis[participant] = contribution
            
            # If parsing failed, provide general analysis
            if not analysis:
                analysis["Overall"] = content
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing participants: {e}")
            return {"Error": "Unable to analyze participant contributions"}
    
    def _extract_action_items(self, transcript: str) -> List[str]:
        """Extract action items and next steps from the discussion"""
        prompt = f"""
Analyze the following discussion transcript and extract any action items, tasks, or next steps that were mentioned.
Return them as a simple list, one item per line. If no action items are found, return "No specific action items identified".

Transcript:
{transcript}

Action Items:"""
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            
            content = response['message']['content'].strip()
            
            if "no action items" in content.lower() or "no specific action" in content.lower():
                return ["No specific action items identified"]
            
            # Parse the response into a list
            action_items = [item.strip('- ').strip() for item in content.split('\n') if item.strip()]
            return action_items[:10]  # Limit to 10 items
            
        except Exception as e:
            self.logger.error(f"Error extracting action items: {e}")
            return ["Unable to extract action items"]
    
    def _analyze_sentiment(self, transcript: str) -> str:
        """Analyze the overall sentiment of the discussion"""
        prompt = f"""
Analyze the overall sentiment and tone of the following discussion transcript.
Respond with one word from: Positive, Negative, Neutral, Mixed

Transcript:
{transcript}

Sentiment:"""
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            
            sentiment = response['message']['content'].strip().lower()
            
            # Normalize the response
            if any(word in sentiment for word in ['positive', 'good', 'great', 'excellent']):
                return "Positive"
            elif any(word in sentiment for word in ['negative', 'bad', 'poor', 'difficult']):
                return "Negative"
            elif any(word in sentiment for word in ['mixed', 'varied', 'both']):
                return "Mixed"
            else:
                return "Neutral"
                
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return "Unknown"
    
    def _create_empty_summary(self) -> DiscussionSummary:
        """Create an empty summary for when no transcript is available"""
        return DiscussionSummary(
            summary="No discussion content available to summarize.",
            key_points=["No discussion recorded"],
            participants_contribution={"Info": "No participants detected"},
            action_items=["No action items identified"],
            sentiment="Neutral",
            duration_minutes=0,
            generated_at=datetime.now()
        )
    
    def _create_error_summary(self, error_message: str) -> DiscussionSummary:
        """Create an error summary when summarization fails"""
        return DiscussionSummary(
            summary=f"Unable to generate summary due to: {error_message}",
            key_points=["Summary generation failed"],
            participants_contribution={"Error": error_message},
            action_items=["Check system configuration"],
            sentiment="Unknown",
            duration_minutes=0,
            generated_at=datetime.now()
        )
    
    def _create_fallback_summary(self, transcript: str, discussion_stats: Dict) -> DiscussionSummary:
        """Create a basic summary when Ollama is not available"""
        lines = transcript.strip().split('\n')
        participants = set()
        topics = []
        
        # Basic analysis without AI
        for line in lines:
            if '] Participant' in line and ':' in line:
                # Extract participant
                try:
                    participant_part = line.split('] ')[1].split(':')[0]
                    participants.add(participant_part)
                    
                    # Extract text for basic topic detection
                    text = line.split(':', 1)[1].strip()
                    if len(text) > 10:  # Meaningful content
                        topics.append(text[:50] + "..." if len(text) > 50 else text)
                except:
                    pass
        
        # Create basic summary
        duration = discussion_stats.get('duration_minutes', 0)
        
        summary = f"Discussion involved {len(participants)} participants over {duration} minutes. "
        if topics:
            summary += f"Main topics included: {', '.join(topics[:3])}"
        else:
            summary += "Discussion content was recorded but detailed analysis requires Ollama."
        
        # Basic key points
        key_points = []
        if len(participants) > 1:
            key_points.append(f"Multi-participant discussion with {len(participants)} people")
        if duration > 0:
            key_points.append(f"Duration: {duration} minutes")
        if topics:
            key_points.extend(topics[:3])
        else:
            key_points.append("Detailed content analysis unavailable (Ollama required)")
        
        # Basic participant analysis
        participants_contribution = {}
        word_counts = discussion_stats.get('words_per_participant', {})
        for participant, count in word_counts.items():
            participants_contribution[participant] = f"Contributed {count} words to the discussion"
        
        if not participants_contribution:
            participants_contribution["Info"] = "Install and run Ollama for detailed participant analysis"
        
        return DiscussionSummary(
            summary=summary,
            key_points=key_points,
            participants_contribution=participants_contribution,
            action_items=["Install Ollama for AI-powered analysis and insights"],
            sentiment="Neutral",
            duration_minutes=duration,
            generated_at=datetime.now()
        )
    
    def generate_quick_insights(self, transcript: str) -> Dict[str, str]:
        """Generate quick insights for real-time display"""
        if not transcript.strip():
            return {
                "current_topic": "No discussion detected",
                "engagement_level": "Low",
                "recommendation": "Start the discussion"
            }
        
        prompt = f"""
Based on this recent discussion transcript, provide quick insights in exactly this format:

Current Topic: [main topic being discussed]
Engagement Level: [High/Medium/Low]
Recommendation: [one short suggestion for the moderator]

Recent transcript:
{transcript[-1000:]}  # Last 1000 characters

Insights:"""
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            
            content = response['message']['content'].strip()
            
            # Parse the structured response
            insights = {}
            lines = content.split('\n')
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    insights[key] = value.strip()
            
            # Ensure we have the expected keys
            return {
                "current_topic": insights.get("current_topic", "Unknown"),
                "engagement_level": insights.get("engagement_level", "Unknown"),
                "recommendation": insights.get("recommendation", "Continue monitoring")
            }
            
        except Exception as e:
            self.logger.error(f"Error generating quick insights: {e}")
            return {
                "current_topic": "Analysis unavailable",
                "engagement_level": "Unknown",
                "recommendation": "Check Ollama service"
            }