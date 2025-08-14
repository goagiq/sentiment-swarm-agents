#!/usr/bin/env python3
"""
Process individual intelligence analysis questions using MCP tools.
"""

import json
from datetime import datetime
from pathlib import Path


def process_question(question_obj, processor=None):
    """
    Process a single question using appropriate MCP tools.
    
    Args:
        question_obj: Dictionary containing question information
        processor: WorkingBatchProcessor instance (optional)
    
    Returns:
        Updated question object with results
    """
    try:
        print(f"\n{'='*80}")
        print(f"Processing Question {question_obj['id']}: {question_obj['question']}")
        print(f"Section: {question_obj['section']}")
        print(f"Subsection: {question_obj['subsection']}")
        print(f"{'='*80}")
        
        # Update status
        question_obj['status'] = 'processing'
        question_obj['started_at'] = datetime.now().isoformat()
        
        # Determine appropriate MCP tool
        tool_name, tool_params = get_mcp_tool_for_question(question_obj['question'])
        
        print(f"Using MCP tool: {tool_name}")
        print(f"Tool parameters: {tool_params}")
        
        # Execute MCP tool
        result = None
        
        if tool_name == "process_content":
            result = mcp_Sentiment_process_content(**tool_params)
        elif tool_name == "analyze_sentiment":
            result = mcp_Sentiment_analyze_sentiment(**tool_params)
        elif tool_name == "extract_entities":
            result = mcp_Sentiment_extract_entities(**tool_params)
        elif tool_name == "generate_knowledge_graph":
            result = mcp_Sentiment_generate_knowledge_graph(**tool_params)
        elif tool_name == "query_knowledge_graph":
            result = mcp_Sentiment_query_knowledge_graph(**tool_params)
        elif tool_name == "analyze_business_intelligence":
            result = mcp_Sentiment_analyze_business_intelligence(**tool_params)
        elif tool_name == "translate_content":
            result = mcp_Sentiment_translate_content(**tool_params)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        # Update question with result
        question_obj['status'] = 'completed'
        question_obj['completed_at'] = datetime.now().isoformat()
        question_obj['result'] = result
        question_obj['error'] = None
        
        # Save result to file
        results_dir = Path("working_batch_results")
        results_dir.mkdir(exist_ok=True)
        
        result_file = results_dir / f"question_{question_obj['id']:03d}_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                "question": question_obj,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Question {question_obj['id']} completed successfully!")
        print(f"Result saved to: {result_file}")
        
        # Update processor status if provided
        if processor:
            processor.status['questions'][question_obj['id'] - 1] = question_obj
            processor.status['completed_questions'] = sum(
                1 for q in processor.status['questions'] if q['status'] == 'completed'
            )
            processor.save_status()
        
        return question_obj
        
    except Exception as e:
        print(f"\n❌ Error processing question {question_obj['id']}: {e}")
        question_obj['status'] = 'error'
        question_obj['error'] = str(e)
        question_obj['completed_at'] = datetime.now().isoformat()
        
        # Update processor status if provided
        if processor:
            processor.status['questions'][question_obj['id'] - 1] = question_obj
            processor.save_status()
        
        return question_obj


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
            "text": question,
            "entity_types": None
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
            "text": question,
            "entity_types": None
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


def load_questions():
    """Load questions from the working batch status file."""
    status_file = "working_batch_status.json"
    if Path(status_file).exists():
        with open(status_file, 'r', encoding='utf-8') as f:
            status = json.load(f)
            return status.get('questions', [])
    return []


def display_progress():
    """Display current progress."""
    questions = load_questions()
    if not questions:
        print("No questions found. Run working_batch_processor.py first.")
        return
    
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
    # Load questions
    questions = load_questions()
    
    if questions:
        print(f"✅ Loaded {len(questions)} questions from working_batch_status.json")
        print(f"Use process_question(questions[0]) to process the first question")
        print(f"Use display_progress() to see current status")
        
        # Show first question
        print(f"\n📋 First question:")
        print(f"ID: {questions[0]['id']}")
        print(f"Question: {questions[0]['question']}")
        print(f"Status: {questions[0]['status']}")
    else:
        print("❌ No questions found. Run working_batch_processor.py first.")
