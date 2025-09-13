#!/usr/bin/env python3
"""
Debug transcript parsing
"""

import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lightweight_summarizer import LightweightSummarizer

def debug_parsing():
    """Debug the transcript parsing"""
    print("🔍 Debugging Transcript Parsing")
    print("=" * 40)
    
    mock_transcript = """[2025-09-13 15:10:00] Participant A: Hello everyone, welcome to our project planning meeting.
[2025-09-13 15:10:05] Participant B: Thanks for organizing this. I think we should discuss the timeline first."""
    
    print(f"Original transcript:")
    print(repr(mock_transcript))
    print()
    
    # Split by lines
    lines = mock_transcript.strip().split('\\n')
    print(f"Lines after split:")
    for i, line in enumerate(lines):
        print(f"{i}: {repr(line)}")
    print()
    
    # Try parsing each line
    import re
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        print(f"Parsing line: {repr(line)}")
        match = re.match(r'\\[(.*?)\\]\\s*(.*?):\\s*(.*)', line)
        if match:
            timestamp_str, participant, text = match.groups()
            print(f"  ✅ Parsed: timestamp='{timestamp_str}', participant='{participant}', text='{text}'")
        else:
            print(f"  ❌ Failed to parse")
        print()

if __name__ == "__main__":
    debug_parsing()