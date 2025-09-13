#!/usr/bin/env python3
"""
Test lightweight summarizer speed and functionality
"""

import time
import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lightweight_summarizer import LightweightSummarizer

def test_lightweight_summarizer():
    """Test the lightweight summarizer"""
    print("⚡ Testing Lightweight Discussion Summarizer")
    print("=" * 50)
    
    # Initialize summarizer
    print("\\n1. Initializing summarizer...")
    start_time = time.time()
    summarizer = LightweightSummarizer()
    init_time = time.time() - start_time
    print(f"✅ Initialized in {init_time:.3f} seconds")
    
    # Test with mock transcript
    print("\\n2. Testing with mock discussion...")
    mock_transcript = """[2025-09-13 15:10:00] Participant A: Hello everyone, welcome to our project planning meeting.
[2025-09-13 15:10:05] Participant B: Thanks for organizing this. I think we should discuss the timeline first.
[2025-09-13 15:10:10] Participant A: Absolutely. We have three main deliverables to complete by next month.
[2025-09-13 15:10:15] Participant B: Let's prioritize them based on client requirements. This is really important.
[2025-09-13 15:10:20] Participant A: I will take the lead on the first deliverable. Can you handle the second one?
[2025-09-13 15:10:25] Participant B: Sure, I can do that. We should also schedule a review meeting for next week.
[2025-09-13 15:10:30] Participant A: Great idea! I think this project will be successful.
[2025-09-13 15:10:35] Participant B: I need to check the budget constraints before we finalize anything."""
    
    mock_stats = {
        'duration_minutes': 5,
        'participant_count': 2,
        'total_segments': 8
    }
    
    # Generate summary
    print("\\n3. Generating summary...")
    start_time = time.time()
    summary = summarizer.generate_summary(mock_transcript, mock_stats)
    generation_time = time.time() - start_time
    
    print(f"✅ Summary generated in {generation_time:.3f} seconds")
    
    # Display results
    print("\\n📋 SUMMARY RESULTS:")
    print("=" * 30)
    print(f"📝 Summary: {summary.summary}")
    print(f"\\n🔑 Key Points:")
    for i, point in enumerate(summary.key_points, 1):
        print(f"   {i}. {point}")
    
    print(f"\\n👥 Participants:")
    for participant, contribution in summary.participants_contribution.items():
        print(f"   • {participant}: {contribution}")
    
    print(f"\\n✅ Action Items:")
    for i, item in enumerate(summary.action_items, 1):
        print(f"   {i}. {item}")
    
    print(f"\\n😊 Sentiment: {summary.sentiment}")
    print(f"⏱️ Duration: {summary.duration_minutes} minutes")
    
    # Performance summary
    print("\\n⚡ PERFORMANCE SUMMARY:")
    print("=" * 30)
    print(f"Initialization: {init_time:.3f}s")
    print(f"Summary Generation: {generation_time:.3f}s")
    print(f"Total Time: {init_time + generation_time:.3f}s")
    
    if generation_time < 0.1:
        print("🎉 EXCELLENT: Sub-100ms generation!")
    elif generation_time < 0.5:
        print("✅ VERY FAST: Sub-500ms generation!")
    elif generation_time < 1.0:
        print("✅ FAST: Sub-1s generation!")
    else:
        print("⚠️ Could be faster, but still reasonable")

if __name__ == "__main__":
    test_lightweight_summarizer()