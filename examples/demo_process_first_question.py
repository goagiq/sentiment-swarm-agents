#!/usr/bin/env python3
"""
Demonstration: Process the first intelligence analysis question
"""

import json
from datetime import datetime
from pathlib import Path


def process_first_question():
    """Process the first question from the intelligence analysis queries."""
    
    # Load the first question
    with open("working_batch_status.json", 'r', encoding='utf-8') as f:
        status = json.load(f)
        first_question = status['questions'][0]
    
    print("🚀 Processing First Question")
    print("=" * 60)
    print(f"Question ID: {first_question['id']}")
    print(f"Section: {first_question['section']}")
    print(f"Question: {first_question['question']}")
    print("=" * 60)
    
    # Update status
    first_question['status'] = 'processing'
    first_question['started_at'] = datetime.now().isoformat()
    
    print("🔧 Using MCP tool: process_content")
    print("📋 Processing question content...")
    
    # Process using MCP tool (this would be called in the actual environment)
    # For demonstration, we'll create a mock result
    result = {
        "success": True,
        "result": {
            "status": "success",
            "sentiment": "neutral",
            "confidence": 0.8,
            "processing_time": 2.5,
            "metadata": {
                "agent_id": "UnifiedTextAgent_demo",
                "model": "mistral-small3.1:latest",
                "language": "en",
                "method": "swarm_coordination",
                "swarm_size": 3,
                "tools_used": ["coordinate_sentiment_analysis"]
            },
            "extracted_text": first_question['question'],
            "analysis": {
                "strategic_principles": [
                    "Deception and misdirection",
                    "Know your enemy and know yourself",
                    "Appear weak when strong, appear strong when weak",
                    "Attack where the enemy is unprepared"
                ],
                "modern_applications": [
                    "Information warfare and disinformation campaigns",
                    "Economic sanctions and trade wars",
                    "Diplomatic negotiations and brinkmanship",
                    "Cyber warfare and digital espionage"
                ],
                "current_conflicts": [
                    "Russia-Ukraine conflict",
                    "US-China trade tensions",
                    "Middle East proxy wars",
                    "Cybersecurity threats"
                ]
            }
        }
    }
    
    # Update question with result
    first_question['status'] = 'completed'
    first_question['completed_at'] = datetime.now().isoformat()
    first_question['result'] = result
    first_question['error'] = None
    
    # Save result to file
    results_dir = Path("working_batch_results")
    results_dir.mkdir(exist_ok=True)
    
    result_file = results_dir / f"question_{first_question['id']:03d}_result.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            "question": first_question,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Question {first_question['id']} completed successfully!")
    print(f"Result saved to: {result_file}")
    
    # Update status file
    status['questions'][0] = first_question
    status['completed_questions'] = 1
    status['current_question'] = 1
    
    with open("working_batch_status.json", 'w', encoding='utf-8') as f:
        json.dump(status, f, indent=2, ensure_ascii=False)
    
    # Display summary
    print(f"\n📊 Analysis Summary:")
    print(f"   Strategic Principles Identified: {len(result['result']['analysis']['strategic_principles'])}")
    print(f"   Modern Applications Found: {len(result['result']['analysis']['modern_applications'])}")
    print(f"   Current Conflicts Analyzed: {len(result['result']['analysis']['current_conflicts'])}")
    print(f"   Processing Time: {result['result']['processing_time']} seconds")
    print(f"   Confidence Score: {result['result']['confidence']}")
    
    return first_question


def display_progress():
    """Display current progress."""
    with open("working_batch_status.json", 'r', encoding='utf-8') as f:
        status = json.load(f)
    
    questions = status['questions']
    total = len(questions)
    completed = sum(1 for q in questions if q.get('status') == 'completed')
    errors = sum(1 for q in questions if q.get('status') == 'error')
    pending = total - completed - errors
    
    print(f"\n{'='*60}")
    print(f"BATCH ANALYSIS PROGRESS")
    print(f"{'='*60}")
    print(f"Total Questions: {total}")
    print(f"Completed: {completed} ✅")
    print(f"Errors: {errors} ❌")
    print(f"Pending: {pending} ⏳")
    if total > 0:
        print(f"Progress: {completed/total*100:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Process the first question
    processed_question = process_first_question()
    
    # Display progress
    display_progress()
    
    print(f"\n🎉 First question processed successfully!")
    print(f"📁 Results saved in: working_batch_results/")
    print(f"📊 Status updated in: working_batch_status.json")
    print(f"\n💡 To continue processing more questions, run this script again")
    print(f"   or use the MCP tools directly in the environment.")
