#!/usr/bin/env python3
"""
Test script to verify question extraction from intelligence_analysis_queries.md
"""

import re
import json
from pathlib import Path

def test_question_extraction():
    """Test the question extraction logic."""
    print("🧪 Testing Question Extraction")
    print("=" * 50)
    
    # Read the queries file
    queries_file = "intelligence_analysis_queries.md"
    if not Path(queries_file).exists():
        print(f"❌ {queries_file} not found!")
        return
    
    with open(queries_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract questions using the same logic as the batch processor
    questions = []
    sections = re.split(r'^##\s+', content, flags=re.MULTILINE)
    
    for section in sections[1:]:  # Skip first empty section
        lines = section.strip().split('\n')
        if not lines:
            continue
        
        section_title = lines[0].strip()
        current_subsection = ""
        section_questions = []
        
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
                    "status": "pending"
                }
                
                questions.append(question_obj)
                section_questions.append(question_obj)
        
        print(f"📋 Section '{section_title}': {len(section_questions)} questions")
    
    print(f"\n✅ Total questions extracted: {len(questions)}")
    
    # Display first few questions as examples
    print(f"\n📝 Sample Questions:")
    print("-" * 50)
    for i, q in enumerate(questions[:3], 1):
        print(f"{i}. [{q['section']}] {q['question']}")
    
    # Test tool selection logic
    print(f"\n🔧 Testing Tool Selection:")
    print("-" * 50)
    
    def get_mcp_tool_for_question(question: str):
        """Test tool selection logic."""
        question_lower = question.lower()
        
        if any(keyword in question_lower for keyword in ['strategic principles', 'art of war', 'military strategy']):
            return "query_knowledge_graph"
        elif any(keyword in question_lower for keyword in ['cultural', 'bias', 'assumptions', 'values']):
            return "extract_entities"
        elif any(keyword in question_lower for keyword in ['threat', 'conflict', 'warfare', 'cyber']):
            return "query_knowledge_graph"
        elif any(keyword in question_lower for keyword in ['language', 'translation', 'classical chinese', 'communication']):
            return "process_content"
        elif any(keyword in question_lower for keyword in ['scenario', 'what if', 'historical pattern']):
            return "generate_knowledge_graph"
        elif any(keyword in question_lower for keyword in ['pattern', 'recurring', 'theme']):
            return "query_knowledge_graph"
        elif any(keyword in question_lower for keyword in ['operational', 'intelligence gathering', 'collection']):
            return "extract_entities"
        elif any(keyword in question_lower for keyword in ['deception', 'counterintelligence', 'warning']):
            return "analyze_sentiment"
        elif any(keyword in question_lower for keyword in ['predictive', 'synthesis', 'cross-cultural']):
            return "analyze_business_intelligence"
        else:
            return "analyze_business_intelligence"
    
    for i, q in enumerate(questions[:5], 1):
        tool = get_mcp_tool_for_question(q['question'])
        print(f"{i}. {tool} -> {q['question'][:60]}...")
    
    # Save test results
    test_results = {
        "total_questions": len(questions),
        "sections": list(set(q['section'] for q in questions)),
        "sample_questions": questions[:5]
    }
    
    with open("test_extraction_results.json", 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Test results saved to: test_extraction_results.json")
    print(f"✅ Question extraction test completed successfully!")

if __name__ == "__main__":
    test_question_extraction()
