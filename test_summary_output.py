#!/usr/bin/env python3
"""
Test what the lightweight summarizer actually returns
"""

import sys
import os
sys.path.append('/Users/fandikaikhsan/Documents/projects/ai-moderator')

from lightweight_summarizer import LightweightSummarizer

# Create test transcript similar to what we saw in debug
test_transcript = """[2024-09-13 16:03:32] Participant 1: Hello everyone, my name is Andika
[2024-09-13 16:03:35] Participant 2: Thank you, Anika
[2024-09-13 16:03:38] Participant 1: I want to take notes this meeting
[2024-09-13 16:03:41] Participant 1: I want to share about the progress of my UX research
[2024-09-13 16:03:44] Participant 1: I found that the hijab is half-caroled in his team"""

test_stats = {
    'duration_minutes': 5.0,
    'total_segments': 5,
    'participants': ['Participant 1', 'Participant 2']
}

print("Testing LightweightSummarizer...")
summarizer = LightweightSummarizer()

print(f"\nInput transcript:\n{test_transcript}")
print(f"\nInput stats: {test_stats}")

summary = summarizer.generate_summary(test_transcript, test_stats)

print(f"\nGenerated summary type: {type(summary)}")
print(f"Summary object: {summary}")

print(f"\nSummary fields:")
print(f"- summary: {summary.summary}")
print(f"- key_points: {summary.key_points}")
print(f"- participants_contribution: {summary.participants_contribution}")
print(f"- action_items: {summary.action_items}")
print(f"- sentiment: {summary.sentiment}")
print(f"- duration_minutes: {summary.duration_minutes}")