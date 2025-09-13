#!/usr/bin/env python3
"""
Test Ollama integration for discussion summarization
"""

import ollama
import sys

def test_ollama_connection():
    """Test if Ollama is running and has required models"""
    print("🔍 Testing Ollama Integration...")
    print("=" * 40)
    
    try:
        # Test connection
        models = ollama.list()
        available_models = [model['name'] for model in models.get('models', [])]
        
        print(f"📊 Ollama is running")
        print(f"📋 Available models: {len(available_models)}")
        
        if available_models:
            print("   Models found:")
            for model in available_models:
                print(f"   • {model}")
        else:
            print("   ⚠️  No models found")
        
        # Test if common models are available
        recommended_models = ['llama3.2', 'llama2', 'llama3', 'mistral']
        working_model = None
        
        for model in recommended_models:
            if any(model in available for available in available_models):
                working_model = model
                break
        
        if working_model:
            print(f"\n✅ Found recommended model: {working_model}")
            
            # Test a simple chat
            print("🧪 Testing chat functionality...")
            response = ollama.chat(
                model=working_model,
                messages=[
                    {
                        'role': 'user',
                        'content': 'Say "Hello from AI Moderator!" in exactly those words.'
                    }
                ]
            )
            
            reply = response['message']['content']
            print(f"💬 Model response: {reply}")
            
            if "Hello from AI Moderator!" in reply:
                print("✅ Ollama integration working perfectly!")
                return True, working_model
            else:
                print("⚠️  Model responded but with unexpected content")
                return True, working_model
                
        else:
            print("\n❌ No recommended models found")
            print("💡 To install a model, run:")
            print("   ollama pull llama3.2")
            print("   or")
            print("   ollama pull llama2")
            return False, None
            
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("\n💡 To fix this:")
        print("1. Install Ollama: https://ollama.ai/")
        print("2. Start Ollama service")
        print("3. Pull a model: ollama pull llama3.2")
        return False, None

def test_discussion_summarizer():
    """Test the discussion summarizer with a sample"""
    print("\n🧪 Testing Discussion Summarizer...")
    print("=" * 40)
    
    try:
        from discussion_summarizer import DiscussionSummarizer
        
        # Test with sample discussion
        sample_transcript = """
[14:30:25] Participant 1: Hi everyone, thanks for joining today's meeting about the new project timeline.
[14:30:35] Participant 2: Happy to be here. I have some concerns about the deadline though.
[14:30:45] Participant 1: What specific concerns do you have?
[14:30:55] Participant 2: The development phase seems rushed. We might need an extra week.
[14:31:05] Participant 3: I agree with Participant 2. Quality is important.
[14:31:15] Participant 1: Okay, let's discuss extending the timeline by one week.
[14:31:25] Participant 2: That would be perfect. We should also plan better testing.
[14:31:35] Participant 3: Yes, and maybe add a buffer for bug fixes.
"""
        
        sample_stats = {
            'duration_minutes': 2.5,
            'words_per_participant': {
                'Participant 1': 25,
                'Participant 2': 30,
                'Participant 3': 15
            }
        }
        
        summarizer = DiscussionSummarizer()
        
        print("📝 Generating summary...")
        summary = summarizer.generate_summary(sample_transcript, sample_stats)
        
        print(f"✅ Summary generated successfully!")
        print(f"📊 Summary length: {len(summary.summary)} characters")
        print(f"🔑 Key points: {len(summary.key_points)}")
        print(f"👥 Participants analyzed: {len(summary.participants_contribution)}")
        print(f"📋 Action items: {len(summary.action_items)}")
        print(f"😊 Sentiment: {summary.sentiment}")
        
        return True
        
    except Exception as e:
        print(f"❌ Discussion summarizer test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 AI Discussion Moderator - Ollama Integration Test")
    print("=" * 50)
    
    # Test Ollama connection
    ollama_works, model = test_ollama_connection()
    
    if ollama_works:
        # Test discussion summarizer
        summarizer_works = test_discussion_summarizer()
        
        print("\n" + "=" * 50)
        print("📊 FINAL RESULTS:")
        print(f"  Ollama Connection: {'✅ Working' if ollama_works else '❌ Failed'}")
        if model:
            print(f"  Best Model: {model}")
        print(f"  Discussion Summarizer: {'✅ Working' if summarizer_works else '❌ Failed'}")
        
        if ollama_works and summarizer_works:
            print("\n🎉 All systems ready! You can now use:")
            print("   • Real-time speech transcription")
            print("   • AI-powered discussion summaries")
            print("   • Participant analysis")
            print("\n🚀 Run the app with: ./run.sh")
        else:
            print("\n⚠️  Some features may not work properly")
    else:
        print("\n❌ Ollama is required for summarization features")
        print("The app will still work for face detection and basic transcription")