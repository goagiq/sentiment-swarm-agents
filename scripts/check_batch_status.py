#!/usr/bin/env python3
"""
Batch Analysis Status Viewer
Displays current progress and results of batch intelligence analysis.
"""

import json
import os
from pathlib import Path
from typing import Dict


def load_status() -> Dict:
    """Load the batch analysis status."""
    status_file = "batch_analysis_status.json"
    if not os.path.exists(status_file):
        print("❌ No batch analysis status found!")
        print("Run the batch analysis first.")
        return {}
    
    try:
        with open(status_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading status: {e}")
        return {}


def display_overall_progress(status: Dict):
    """Display overall progress statistics."""
    if not status:
        return
    
    total = len(status.get('questions', []))
    completed = sum(1 for q in status.get('questions', [])
                    if q.get('status') == 'completed')
    errors = sum(1 for q in status.get('questions', [])
                 if q.get('status') == 'error')
    pending = total - completed - errors
    
    print("\n" + "="*60)
    print("📊 BATCH ANALYSIS STATUS")
    print("="*60)
    print(f"Start Time: {status.get('start_time', 'Unknown')}")
    print(f"Total Questions: {total}")
    print(f"Completed: {completed} ✅")
    print(f"Errors: {errors} ❌")
    print(f"Pending: {pending} ⏳")
    if total > 0:
        print(f"Progress: {completed/total*100:.1f}%")
    print(f"{'='*60}")


def display_question_details(status: Dict, show_all: bool = False):
    """Display detailed question information."""
    if not status or 'questions' not in status:
        return
    
    questions = status['questions']
    
    print("\n📋 QUESTION DETAILS")
    print("="*80)
    
    for i, question in enumerate(questions, 1):
        status_icon = {
            'completed': '✅',
            'error': '❌',
            'processing': '🔄',
            'pending': '⏳'
        }.get(question.get('status', 'pending'), '❓')
        
        print(f"\n{status_icon} Question {question.get('id', i)}")
        print(f"   Section: {question.get('section', 'Unknown')}")
        print(f"   Subsection: {question.get('subsection', 'N/A')}")
        print(f"   Status: {question.get('status', 'pending')}")
        
        if question.get('started_at'):
            print(f"   Started: {question.get('started_at')}")
        if question.get('completed_at'):
            print(f"   Completed: {question.get('completed_at')}")
        if question.get('error'):
            print(f"   Error: {question.get('error')}")
        
        if show_all or question.get('status') in ['completed', 'error']:
            print(f"   Question: {question.get('question', 'N/A')}")
        
        if question.get('status') == 'completed' and question.get('result'):
            result = question.get('result', {})
            if isinstance(result, dict):
                print(f"   Result Type: {result.get('type', 'Unknown')}")
                print(f"   Result Summary: {str(result)[:100]}...")
            else:
                print(f"   Result: {str(result)[:100]}...")


def display_recent_results(status: Dict, limit: int = 5):
    """Display recent completed results."""
    if not status or 'questions' not in status:
        return
    
    completed_questions = [
        q for q in status['questions'] 
        if q.get('status') == 'completed'
    ]
    
    if not completed_questions:
        print("\n📝 No completed questions yet.")
        return
    
    print(f"\n📝 RECENT RESULTS (Last {limit})")
    print(f"{'='*80}")
    
    # Sort by completion time
    completed_questions.sort(
        key=lambda x: x.get('completed_at', ''), 
        reverse=True
    )
    
    for question in completed_questions[:limit]:
        print(f"\n✅ Question {question.get('id')}")
        print(f"   Question: {question.get('question', 'N/A')}")
        print(f"   Completed: {question.get('completed_at', 'Unknown')}")
        
        result = question.get('result', {})
        if isinstance(result, dict):
            print(f"   Result Type: {result.get('type', 'Unknown')}")
            if 'content' in result:
                print(f"   Content Preview: {result['content'][:150]}...")
        else:
            print(f"   Result Preview: {str(result)[:150]}...")


def list_result_files():
    """List available result files."""
    results_dir = Path("batch_analysis_results")
    if not results_dir.exists():
        print("\n📁 No results directory found.")
        return
    
    result_files = list(results_dir.glob("question_*_result.json"))
    if not result_files:
        print("\n📁 No result files found.")
        return
    
    print(f"\n📁 AVAILABLE RESULT FILES ({len(result_files)})")
    print(f"{'='*60}")
    
    for result_file in sorted(result_files):
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                question = data.get('question', {})
                print(f"   {result_file.name}")
                print(f"      Question ID: {question.get('id', 'Unknown')}")
                print(f"      Status: {question.get('status', 'Unknown')}")
                print(f"      Section: {question.get('section', 'Unknown')}")
        except Exception as e:
            print(f"   {result_file.name} (Error reading: {e})")


def main():
    """Main function to display batch analysis status."""
    print("🔍 Batch Analysis Status Viewer")
    print("=" * 50)
    
    # Load status
    status = load_status()
    
    if not status:
        return
    
    # Display overall progress
    display_overall_progress(status)
    
    # Display recent results
    display_recent_results(status, limit=3)
    
    # List result files
    list_result_files()
    
    # Ask user what they want to see
    print(f"\n{'='*60}")
    print("Options:")
    print("1. Show all question details")
    print("2. Show only completed questions")
    print("3. Show only error questions")
    print("4. Exit")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            display_question_details(status, show_all=True)
        elif choice == "2":
            completed = [q for q in status.get('questions', []) 
                        if q.get('status') == 'completed']
            if completed:
                print(f"\n✅ COMPLETED QUESTIONS ({len(completed)})")
                print(f"{'='*80}")
                for q in completed:
                    print(f"\nQuestion {q.get('id')}: {q.get('question')}")
                    print(f"Section: {q.get('section')}")
                    print(f"Completed: {q.get('completed_at')}")
            else:
                print("\nNo completed questions yet.")
        elif choice == "3":
            errors = [q for q in status.get('questions', []) 
                     if q.get('status') == 'error']
            if errors:
                print(f"\n❌ ERROR QUESTIONS ({len(errors)})")
                print(f"{'='*80}")
                for q in errors:
                    print(f"\nQuestion {q.get('id')}: {q.get('question')}")
                    print(f"Error: {q.get('error')}")
            else:
                print("\nNo error questions found.")
        elif choice == "4":
            print("Goodbye!")
        else:
            print("Invalid choice.")
            
    except KeyboardInterrupt:
        print("\n\nGoodbye!")


if __name__ == "__main__":
    main()
