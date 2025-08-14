#!/usr/bin/env python3
"""
Batch Intelligence Analysis Processor
Processes all questions from intelligence_analysis_queries.md systematically
with status tracking and user interaction between questions.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Import MCP functions
try:
    from mcp_Sentiment import (
        process_content as mcp_Sentiment_process_content,
        analyze_sentiment as mcp_Sentiment_analyze_sentiment,
        extract_entities as mcp_Sentiment_extract_entities,
        generate_knowledge_graph as mcp_Sentiment_generate_knowledge_graph,
        query_knowledge_graph as mcp_Sentiment_query_knowledge_graph,
        analyze_business_intelligence as mcp_Sentiment_analyze_business_intelligence,
        translate_content as mcp_Sentiment_translate_content
    )
except ImportError:
    # Fallback to using the MCP client if direct import fails
    from src.core.unified_mcp_client import UnifiedMCPClient
    
    # Create mock functions that use the MCP client
    async def mcp_Sentiment_process_content(**kwargs):
        client = UnifiedMCPClient()
        return await client.call_tool("process_content", kwargs)
    
    async def mcp_Sentiment_analyze_sentiment(**kwargs):
        client = UnifiedMCPClient()
        return await client.call_tool("analyze_sentiment", kwargs)
    
    async def mcp_Sentiment_extract_entities(**kwargs):
        client = UnifiedMCPClient()
        return await client.call_tool("extract_entities", kwargs)
    
    async def mcp_Sentiment_generate_knowledge_graph(**kwargs):
        client = UnifiedMCPClient()
        return await client.call_tool("generate_knowledge_graph", kwargs)
    
    async def mcp_Sentiment_query_knowledge_graph(**kwargs):
        client = UnifiedMCPClient()
        return await client.call_tool("query_knowledge_graph", kwargs)
    
    async def mcp_Sentiment_analyze_business_intelligence(**kwargs):
        client = UnifiedMCPClient()
        return await client.call_tool("analyze_business_intelligence", kwargs)
    
    async def mcp_Sentiment_translate_content(**kwargs):
        client = UnifiedMCPClient()
        return await client.call_tool("translate_content", kwargs)


class IntelligenceAnalysisBatchProcessor:
    def __init__(self, queries_file: str = "intelligence_analysis_queries.md"):
        self.queries_file = queries_file
        self.status_file = "batch_analysis_status.json"
        self.results_dir = Path("batch_analysis_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Load or initialize status
        self.status = self.load_status()
        
    def load_status(self) -> Dict:
        """Load existing status or create new one."""
        if os.path.exists(self.status_file):
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
                "content_type": "text",
                "language": "en"
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
    
    async def process_question(self, question_obj: Dict) -> Dict:
        """Process a single question using appropriate MCP tools."""
        try:
            print(f"\n{'='*80}")
            print(f"Processing Question {question_obj['id']}: {question_obj['question']}")
            print(f"Section: {question_obj['section']}")
            print(f"Subsection: {question_obj['subsection']}")
            print(f"{'='*80}")
            
            # Update status
            question_obj['status'] = 'processing'
            question_obj['started_at'] = datetime.now().isoformat()
            self.save_status()
            
            # Determine appropriate MCP tool
            tool_name, tool_params = self.get_mcp_tool_for_question(question_obj['question'])
            
            print(f"Using MCP tool: {tool_name}")
            print(f"Tool parameters: {tool_params}")
            
            # Execute MCP tool
            result = None
            
            if tool_name == "process_content":
                result = await mcp_Sentiment_process_content(**tool_params)
            elif tool_name == "analyze_sentiment":
                result = await mcp_Sentiment_analyze_sentiment(**tool_params)
            elif tool_name == "extract_entities":
                result = await mcp_Sentiment_extract_entities(**tool_params)
            elif tool_name == "generate_knowledge_graph":
                result = await mcp_Sentiment_generate_knowledge_graph(**tool_params)
            elif tool_name == "query_knowledge_graph":
                result = await mcp_Sentiment_query_knowledge_graph(**tool_params)
            elif tool_name == "analyze_business_intelligence":
                result = await mcp_Sentiment_analyze_business_intelligence(**tool_params)
            elif tool_name == "translate_content":
                result = await mcp_Sentiment_translate_content(**tool_params)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            # Update question with result
            question_obj['status'] = 'completed'
            question_obj['completed_at'] = datetime.now().isoformat()
            question_obj['result'] = result
            question_obj['error'] = None
            
            # Save result to file
            result_file = self.results_dir / f"question_{question_obj['id']:03d}_result.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "question": question_obj,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ Question {question_obj['id']} completed successfully!")
            print(f"Result saved to: {result_file}")
            
            return question_obj
            
        except Exception as e:
            print(f"\n❌ Error processing question {question_obj['id']}: {e}")
            question_obj['status'] = 'error'
            question_obj['error'] = str(e)
            question_obj['completed_at'] = datetime.now().isoformat()
            return question_obj
    
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
    
    async def run_batch_analysis(self):
        """Run the complete batch analysis."""
        print("🚀 Starting Batch Intelligence Analysis")
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
        
        # Process questions
        for i, question_obj in enumerate(self.status['questions']):
            # Skip completed questions
            if question_obj['status'] in ['completed', 'error']:
                continue
            
            # Display current question
            self.display_question_summary(question_obj)
            print(f"\n❓ Question: {question_obj['question']}")
            
            # Process the question
            updated_question = await self.process_question(question_obj)
            
            # Update status
            self.status['questions'][i] = updated_question
            self.status['completed_questions'] = sum(1 for q in self.status['questions'] if q['status'] == 'completed')
            self.status['current_question'] = i + 1
            self.save_status()
            
            # Display updated progress
            self.display_progress()
            
            # Wait for user input before proceeding
            if i < len(self.status['questions']) - 1:  # Not the last question
                print(f"\n⏸️  Question {question_obj['id']} completed. Press Enter to continue to the next question...")
                input()
        
        print(f"\n🎉 Batch analysis completed!")
        print(f"Total questions processed: {self.status['completed_questions']}")
        print(f"Results saved in: {self.results_dir}")
        print(f"Status saved in: {self.status_file}")


async def main():
    """Main function to run the batch analysis."""
    processor = IntelligenceAnalysisBatchProcessor()
    await processor.run_batch_analysis()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
