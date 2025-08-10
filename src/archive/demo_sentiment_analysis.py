#!/usr/bin/env python3
"""
Demonstration script for the Sentiment Analysis Swarm system.
Shows various examples of sentiment analysis capabilities.
"""

import requests
from typing import Dict, Any


def analyze_text(text: str, language: str = "en") -> Dict[str, Any]:
    """Analyze text sentiment using the API."""
    url = "http://localhost:8000/analyze/text"
    payload = {
        "content": text,
        "language": language,
        "reflection_enabled": True,
        "max_iterations": 3,
        "confidence_threshold": 0.8
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error analyzing text: {e}")
        return {}


def print_analysis_result(text: str, result: Dict[str, Any]):
    """Print analysis results in a formatted way."""
    if not result:
        print("❌ Analysis failed")
        return
    
    sentiment = result.get('sentiment', {})
    metadata = result.get('metadata', {})
    
    print(f"📝 Text: '{text[:80]}{'...' if len(text) > 80 else ''}'")
    print(f"🎯 Sentiment: {sentiment.get('label', 'unknown')}")
    print(f"📊 Confidence: {sentiment.get('confidence', 0):.2f}")
    print(f"🤖 Agent: {metadata.get('agent_id', 'unknown')}")
    print(f"⏱️  Processing Time: {metadata.get('processing_time', 0):.2f}s")
    
    if 'reflection' in result:
        reflection = result['reflection']
        print(f"🔄 Reflection Iterations: {len(reflection)}")
        for i, iteration in enumerate(reflection, 1):
            confidence = iteration.get('confidence', 0)
            print(f"   Iteration {i}: confidence {confidence:.2f}")
    
    print("-" * 60)


def main():
    """Run the demonstration."""
    print("🚀 Sentiment Analysis Swarm - Demonstration")
    print("=" * 60)
    
    # Test cases with different sentiment types
    test_cases = [
        {
            "text": "I absolutely love this new smartphone! The camera quality is incredible and the battery life is amazing. Best purchase I've ever made!",
            "description": "Highly positive review"
        },
        {
            "text": "This product is absolutely terrible. It broke after one day of use. The customer service was horrible and they refused to give me a refund. I'm extremely disappointed.",
            "description": "Highly negative review"
        },
        {
            "text": "The product arrived on time and was packaged well. It functions as described in the specifications. The delivery was standard.",
            "description": "Neutral review"
        },
        {
            "text": "Mixed feelings about this one. The design is beautiful and the features are great, but it's quite expensive and the learning curve is steep. Not sure if it's worth the price.",
            "description": "Mixed/ambiguous sentiment"
        },
        {
            "text": "Wow! This exceeded all my expectations! The quality is outstanding, the performance is lightning fast, and the value for money is incredible. I can't recommend it enough!",
            "description": "Extremely positive"
        }
    ]
    
    print(f"Testing {len(test_cases)} different text samples...\n")
    
    for i, case in enumerate(test_cases, 1):
        print(f"🔍 Test Case {i}: {case['description']}")
        print("=" * 60)
        
        result = analyze_text(case['text'])
        print_analysis_result(case['text'], result)
        
        if i < len(test_cases):
            print()
    
    # Show system status
    print("📊 System Status")
    print("=" * 60)
    
    try:
        health_response = requests.get("http://localhost:8000/health")
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ Status: {health_data['status']}")
            print(f"🤖 Active Agents: {len(health_data['agents'])}")
            print(f"🧠 Available Models: {len(health_data['models'])}")
            
            print("\n🤖 Agent Details:")
            for agent_id, agent_info in health_data['agents'].items():
                agent_type = agent_info['agent_type']
                status = agent_info['status']
                model = agent_info['model']
                print(f"   - {agent_type}: {status} (using {model})")
        else:
            print("❌ Could not retrieve system status")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error checking system status: {e}")
    
    print("\n🎉 Demonstration completed!")
    print("The system successfully analyzed various text samples with different sentiment types.")


if __name__ == "__main__":
    main()
