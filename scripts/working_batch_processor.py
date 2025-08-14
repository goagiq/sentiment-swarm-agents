#!/usr/bin/env python3
"""
Working Batch Intelligence Analysis Processor
Processes questions using MCP tools available in this environment.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


class WorkingBatchProcessor:
    def __init__(self, queries_file: str = "intelligence_analysis_queries.md"):
        self.queries_file = queries_file
        self.status_file = "working_batch_status.json"
        self.results_dir = Path("working_batch_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Load or initialize status
        self.status = self.load_status()
        
    def load_status(self) -> Dict:
        """Load existing status or create new one."""
        if Path(self.status_file).exists():
            try:
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading status file: {e}")
        
        return {
            "start_time": datetime.now().isoformat(),
            "total_questions": 0,
            "completed_questions": 0,
            "current_question": 0,
            "questions": [],
            "results": {}
        }
    
    def save_status(self):
        """Save current status to file."""
        try:
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(self.status, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving status: {e}")
    
    def extract_questions_from_markdown(self) -> List[Dict]:
        """Extract all questions from the markdown file."""
        questions = []
        
        try:
            with open(self.queries_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split content into sections
            sections = re.split(r'^##\s+', content, flags=re.MULTILINE)
            
            for section in sections[1:]:  # Skip first empty section
                lines = section.strip().split('\n')
                if not lines:
                    continue
                
                section_title = lines[0].strip()
                
                # Extract questions from each section
                section_questions = []
                current_subsection = ""
                
                for line in lines[1:]:
                    line = line.strip()
                    
                    # Check for subsections
                    if line.startswith('###'):
                        current_subsection = line.replace('###', '').strip()
                        continue
                    
                    # Extract questions (lines starting with - **)
                    if line.startswith('- **') and line.endswith('**'):
                        question_text = line[4:-2]  # Remove '- **' and '**'
                        
                        question_obj = {
                            "id": len(questions) + 1,
                            "section": section_title,
                            "subsection": current_subsection,
                            "question": question_text,
                            "status": "pending",
                            "started_at": None,
                            "completed_at": None,
                            "result": None,
                            "error": None
                        }
                        
                        questions.append(question_obj)
                        section_questions.append(question_obj)
                
                print(f"Extracted {len(section_questions)} questions from section: {section_title}")
            
        except Exception as e:
            print(f"Error extracting questions: {e}")
            return []
        
        return questions
    
    def get_mcp_tool_for_question(self, question: str) -> Tuple[str, Dict]:
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
    
    def display_progress(self):
        """Display current progress."""
        total = len(self.status['questions'])
        completed = sum(1 for q in self.status['questions'] if q['status'] == 'completed')
        errors = sum(1 for q in self.status['questions'] if q['status'] == 'error')
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
    
    def display_question_summary(self, question_obj: Dict):
        """Display summary of a question."""
        print(f"\n📋 Question Summary:")
        print(f"   ID: {question_obj['id']}")
        print(f"   Section: {question_obj['section']}")
        print(f"   Subsection: {question_obj['subsection']}")
        print(f"   Status: {question_obj['status']}")
        if question_obj['started_at']:
            print(f"   Started: {question_obj['started_at']}")
        if question_obj['completed_at']:
            print(f"   Completed: {question_obj['completed_at']}")
        if question_obj['error']:
            print(f"   Error: {question_obj['error']}")
    
    def run_batch_analysis(self):
        """Run the complete batch analysis."""
        print("🚀 Starting Working Batch Intelligence Analysis")
        print(f"Queries file: {self.queries_file}")
        print(f"Status file: {self.status_file}")
        print(f"Results directory: {self.results_dir}")
        
        # Extract questions if not already done
        if not self.status['questions']:
            print("\n📖 Extracting questions from markdown file...")
            self.status['questions'] = self.extract_questions_from_markdown()
            self.status['total_questions'] = len(self.status['questions'])
            self.save_status()
        
        if not self.status['questions']:
            print("❌ No questions found in the markdown file!")
            return
        
        # Display initial progress
        self.display_progress()
        
        print(f"\n📝 Ready to process {len(self.status['questions'])} questions.")
        print(f"Each question will be processed using appropriate MCP tools.")
        print(f"Results will be saved to individual JSON files.")
        print(f"Status will be tracked for resuming later.")
        
        return self.status['questions']


def main():
    """Main function to run the batch analysis."""
    processor = WorkingBatchProcessor()
    questions = processor.run_batch_analysis()
    
    if questions:
        print(f"\n✅ Successfully extracted {len(questions)} questions!")
        print(f"Use the process_question() function to process individual questions.")
        print(f"Example: process_question(questions[0])")
        
        # Display first few questions
        print(f"\n📋 First 3 questions:")
        for i, q in enumerate(questions[:3], 1):
            print(f"{i}. [{q['section']}] {q['question']}")
    
    return processor, questions


if __name__ == "__main__":
    processor, questions = main()
