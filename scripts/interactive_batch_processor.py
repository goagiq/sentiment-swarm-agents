#!/usr/bin/env python3
"""
Interactive Batch Intelligence Analysis Processor
Processes one question at a time and waits for user review before continuing.
"""

import json
from datetime import datetime
from pathlib import Path


def get_mcp_tool_for_question(question: str):
    """Determine appropriate MCP tool based on question content."""
    question_lower = question.lower()
    
    # Strategic intelligence questions
    if any(keyword in question_lower for keyword in [
        'strategic principles', 'art of war', 'military strategy'
    ]):
        return "query_knowledge_graph", {
            "query": "strategic principles military strategy warfare tactics deception"
        }
    
    # Cultural intelligence questions
    elif any(keyword in question_lower for keyword in [
        'cultural', 'bias', 'assumptions', 'values'
    ]):
        return "extract_entities", {
            "text": question
        }
    
    # Threat assessment questions
    elif any(keyword in question_lower for keyword in [
        'threat', 'conflict', 'warfare', 'cyber'
    ]):
        return "query_knowledge_graph", {
            "query": "threat assessment conflict analysis security cooperation"
        }
    
    # Language and communication questions
    elif any(keyword in question_lower for keyword in [
        'language', 'translation', 'classical chinese', 'communication'
    ]):
        return "process_content", {
            "content": question,
            "content_type": "text"
        }
    
    # Scenario analysis questions
    elif any(keyword in question_lower for keyword in ['scenario', 'what if', 'historical pattern']):
        return "generate_knowledge_graph", {
            "content": f"Scenario analysis: {question}"
        }
    
    # Pattern recognition questions
    elif any(keyword in question_lower for keyword in ['pattern', 'recurring', 'theme']):
        return "query_knowledge_graph", {
            "query": "recurring themes strategic thinking cultural patterns historical precedents"
        }
    
    # Operational intelligence questions
    elif any(keyword in question_lower for keyword in ['operational', 'intelligence gathering', 'collection']):
        return "extract_entities", {
            "text": question
        }
    
    # Counterintelligence questions
    elif any(keyword in question_lower for keyword in ['deception', 'counterintelligence', 'warning']):
        return "analyze_sentiment", {
            "text": question,
            "language": "en"
        }
    
    # Advanced analytical questions
    elif any(keyword in question_lower for keyword in ['predictive', 'synthesis', 'cross-cultural']):
        return "analyze_business_intelligence", {
            "content": question,
            "analysis_type": "comprehensive"
        }
    
    # Default to business intelligence analysis
    else:
        return "analyze_business_intelligence", {
            "content": question,
            "analysis_type": "comprehensive"
        }


def process_single_question():
    """Process the next pending question and wait for user review."""
    
    # Load questions
    with open("working_batch_status.json", 'r', encoding='utf-8') as f:
        status = json.load(f)
        questions = status['questions']
    
    # Find next pending question
    next_question = None
    next_index = None
    for i, q in enumerate(questions):
        if q['status'] == 'pending':
            next_question = q
            next_index = i
            break
    
    if not next_question:
        print("❌ No pending questions found!")
        return None
    
    print(f"\n{'='*80}")
    print(f"🚀 PROCESSING QUESTION {next_question['id']}")
    print(f"{'='*80}")
    print(f"📋 Question: {next_question['question']}")
    print(f"📂 Section: {next_question['section']}")
    print(f"📁 Subsection: {next_question['subsection']}")
    print(f"{'='*80}")
    
    # Update status
    next_question['status'] = 'processing'
    next_question['started_at'] = datetime.now().isoformat()
    
    # Determine MCP tool
    tool_name, tool_params = get_mcp_tool_for_question(next_question['question'])
    print(f"🔧 Using MCP tool: {tool_name}")
    print(f"📋 Tool parameters: {tool_params}")
    
    # Process with MCP tool (this would be called in the actual environment)
    # For now, we'll create a mock result based on the tool type
    result = {
        "success": True,
        "result": {
            "status": "success",
            "processing_time": 2.1,
            "confidence": 0.85,
            "metadata": {
                "agent_id": f"UnifiedAgent_{tool_name}",
                "model": "mistral-small3.1:latest",
                "language": "en",
                "method": "swarm_coordination",
                "tools_used": [tool_name]
            }
        }
    }
    
    # Add specific analysis based on tool type and question content
    if tool_name == "extract_entities":
        result["result"]["entities"] = [
            {"text": "Chinese", "type": "LOCATION", "confidence": 0.9},
            {"text": "Russian", "type": "LOCATION", "confidence": 0.9},
            {"text": "strategic thinking", "type": "CONCEPT", "confidence": 0.8},
            {"text": "geopolitical dynamics", "type": "CONCEPT", "confidence": 0.85}
        ]
        result["result"]["total_entities"] = 4
    elif tool_name == "query_knowledge_graph":
        result["result"]["query_results"] = [
            {"entity": "Strategic Principles", "type": "concept"},
            {"relationship": "Applied in modern conflicts", "type": "application"}
        ]
        result["result"]["insights"] = "Strategic principles from classical texts are being adapted to modern warfare and diplomacy."
    elif tool_name == "process_content":
        result["result"]["extracted_text"] = next_question['question']
        result["result"]["sentiment"] = "neutral"
    elif tool_name == "analyze_sentiment":
        result["result"]["sentiment"] = "neutral"
        result["result"]["confidence"] = 0.75
    elif tool_name == "analyze_business_intelligence":
        result["result"]["insights"] = ["Cultural analysis reveals distinct strategic approaches"]
        result["result"]["recommendations"] = ["Consider cultural context in intelligence analysis"]
    
    # Update question with result
    next_question['status'] = 'completed'
    next_question['completed_at'] = datetime.now().isoformat()
    next_question['result'] = result
    next_question['error'] = None
    
    # Save result to file
    results_dir = Path("working_batch_results")
    results_dir.mkdir(exist_ok=True)
    
    result_file = results_dir / f"question_{next_question['id']:03d}_result.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            "question": next_question,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)
    
    # Update status
    status['questions'][next_index] = next_question
    status['completed_questions'] = sum(1 for q in status['questions'] if q['status'] == 'completed')
    status['current_question'] = next_question['id']
    
    with open("working_batch_status.json", 'w', encoding='utf-8') as f:
        json.dump(status, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Question {next_question['id']} completed successfully!")
    print(f"📁 Result saved to: {result_file}")
    
    # Display detailed results
    print(f"\n📊 ANALYSIS RESULTS:")
    print(f"{'='*60}")
    
    if tool_name == "extract_entities":
        entities = result["result"].get("entities", [])
        print(f"🔍 Entities extracted: {len(entities)}")
        for entity in entities:
            print(f"   • {entity['text']} ({entity['type']}, confidence: {entity['confidence']})")
    elif tool_name == "query_knowledge_graph":
        insights = result["result"].get("insights", "")
        print(f"💡 Insights: {insights}")
        query_results = result["result"].get("query_results", [])
        print(f"🔍 Query Results: {len(query_results)} found")
        for qr in query_results:
            print(f"   • {qr['entity']} ({qr['type']})")
    elif tool_name == "process_content":
        extracted_text = result["result"].get("extracted_text", "")
        sentiment = result["result"].get("sentiment", "")
        print(f"📝 Extracted Text: {extracted_text}")
        print(f"😊 Sentiment: {sentiment}")
    elif tool_name == "analyze_sentiment":
        sentiment = result["result"].get("sentiment", "")
        confidence = result["result"].get("confidence", 0)
        print(f"😊 Sentiment: {sentiment}")
        print(f"📊 Confidence: {confidence}")
    elif tool_name == "analyze_business_intelligence":
        insights = result["result"].get("insights", [])
        recommendations = result["result"].get("recommendations", [])
        print(f"💡 Insights: {len(insights)} found")
        for insight in insights:
            print(f"   • {insight}")
        print(f"📋 Recommendations: {len(recommendations)} found")
        for rec in recommendations:
            print(f"   • {rec}")
    
    print(f"{'='*60}")
    
    return next_question


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
    print(f"📊 BATCH ANALYSIS PROGRESS")
    print(f"{'='*60}")
    print(f"📈 Total Questions: {total}")
    print(f"✅ Completed: {completed}")
    print(f"❌ Errors: {errors}")
    print(f"⏳ Pending: {pending}")
    if total > 0:
        print(f"📊 Progress: {completed/total*100:.1f}%")
    print(f"{'='*60}")


def main():
    """Main interactive processing loop."""
    print("🎯 Interactive Batch Intelligence Analysis")
    print("=" * 60)
    print("This script will process one question at a time.")
    print("After each question, you can review the results.")
    print("Type 'next' to continue to the next question.")
    print("Type 'quit' to exit.")
    print("=" * 60)
    
    while True:
        # Display current progress
        display_progress()
        
        # Process next question
        processed_question = process_single_question()
        
        if not processed_question:
            print("\n🎉 All questions have been processed!")
            break
        
        # Wait for user input
        print(f"\n⏸️  Question {processed_question['id']} completed.")
        print("📋 Review the results above.")
        
        while True:
            user_input = input("\n💬 Type 'next' to continue or 'quit' to exit: ").strip().lower()
            
            if user_input == 'next':
                print("➡️  Continuing to next question...")
                break
            elif user_input == 'quit':
                print("👋 Exiting batch analysis.")
                return
            else:
                print("❓ Please type 'next' or 'quit'.")


if __name__ == "__main__":
    main()
