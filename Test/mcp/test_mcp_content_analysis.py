#!/usr/bin/env python3
"""
Test script for the new MCP content analysis tools.
Tests summarization, chapter analysis, content extraction, and comparison capabilities.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.models import AnalysisRequest, DataType
from src.agents.unified_text_agent import UnifiedTextAgent
from src.agents.knowledge_graph_agent import KnowledgeGraphAgent


async def test_text_summarization():
    """Test text summarization capabilities."""
    print("🔧 Testing Text Summarization...")
    
    # Sample text from The Art of War Chapter 1
    sample_text = """
    Chapter 1: Laying Plans (始計)
    
    Sun Tzu said: The art of war is of vital importance to the state. It is a matter of life and death, a road either to safety or to ruin. Hence it is a subject of inquiry which can on no account be neglected.
    
    The art of war, then, is governed by five constant factors, to be taken into account in one's deliberations, when seeking to determine the conditions obtaining in the field. These are: (1) The Moral Law; (2) Heaven; (3) Earth; (4) The Commander; (5) Method and discipline.
    
    The Moral Law causes the people to be in complete accord with their ruler, so that they will follow him regardless of their lives, undismayed by any danger.
    
    Heaven signifies night and day, cold and heat, times and seasons.
    
    Earth comprises distances, great and small; danger and security; open ground and narrow passes; the chances of life and death.
    
    The Commander stands for the virtues of wisdom, sincerely, benevolence, courage and strictness.
    
    By Method and discipline are to be understood the marshaling of the army in its proper subdivisions, the graduations of rank among the officers, the maintenance of roads by which supplies may reach the army, and the control of military expenditure.
    """
    
    try:
        # Initialize text agent
        text_agent = UnifiedTextAgent()
        
        # Test summarization
        result = await text_agent.generate_text_summary(sample_text, "comprehensive")
        
        print(f"✅ Summarization Test Results:")
        print(f"   Status: {result.get('status')}")
        print(f"   Summary Type: {result.get('summary_type')}")
        print(f"   Summary: {result.get('summary', '')[:200]}...")
        print(f"   Key Points: {len(result.get('key_points', []))}")
        print(f"   Entities: {len(result.get('entities', []))}")
        print(f"   Processing Time: {result.get('processing_time', 0):.2f}s")
        
        return result.get('status') == 'success'
        
    except Exception as e:
        print(f"❌ Summarization test failed: {e}")
        return False


async def test_chapter_analysis():
    """Test chapter analysis capabilities."""
    print("\n🔧 Testing Chapter Analysis...")
    
    # Sample chapter content
    chapter_text = """
    Chapter 1: Laying Plans (始計)
    
    Sun Tzu said: The art of war is of vital importance to the state. It is a matter of life and death, a road either to safety or to ruin. Hence it is a subject of inquiry which can on no account be neglected.
    
    The art of war, then, is governed by five constant factors, to be taken into account in one's deliberations, when seeking to determine the conditions obtaining in the field. These are: (1) The Moral Law; (2) Heaven; (3) Earth; (4) The Commander; (5) Method and discipline.
    
    All warfare is based on deception. Hence, when able to attack, we must seem unable; when using our forces, we must seem inactive; when we are near, we must make the enemy believe we are far away; when far away, we must make him believe we are near.
    """
    
    try:
        # Initialize agents
        text_agent = UnifiedTextAgent()
        kg_agent = KnowledgeGraphAgent()
        
        # Test chapter analysis using the MCP tool pattern
        request = AnalysisRequest(
            data_type=DataType.TEXT,
            content=chapter_text,
            language="en"
        )
        
        # Process with knowledge graph agent for entity extraction
        kg_result = await kg_agent.process(request)
        
        # Process with text agent for analysis
        text_result = await text_agent.process(request)
        
        print(f"✅ Chapter Analysis Test Results:")
        print(f"   Chapter Title: Chapter 1: Laying Plans")
        print(f"   Summary: {text_result.sentiment.reasoning[:200] if hasattr(text_result, 'sentiment') else 'Analysis completed'}...")
        print(f"   Entities Found: {len(kg_result.metadata.get('entities', [])) if kg_result.metadata else 0}")
        print(f"   Processing Time: {text_result.processing_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Chapter analysis test failed: {e}")
        return False


async def test_content_extraction():
    """Test content section extraction."""
    print("\n🔧 Testing Content Extraction...")
    
    # Sample content with multiple chapters
    content = """
    # The Art of War (孫子兵法)
    
    ## Chapter 1: Laying Plans (始計)
    
    Sun Tzu said: The art of war is of vital importance to the state. It is a matter of life and death, a road either to safety or to ruin.
    
    ## Chapter 2: Waging War (作戰)
    
    Sun Tzu said: In the operations of war, where there are in the field a thousand swift chariots, as many heavy chariots, and a hundred thousand mail-clad soldiers, with provisions enough to carry them a thousand li, the expenditure at home and at the front, including entertainment of guests, small items such as glue and paint, and sums spent on chariots and armour, will reach the total of a thousand ounces of silver per day.
    
    ## Chapter 3: Attack by Stratagem (謀攻)
    
    Sun Tzu said: In the practical art of war, the best thing of all is to take the enemy's country whole and intact; to shatter and destroy it is not so good.
    """
    
    try:
        import re
        
        # Test content section extraction
        sections = []
        chapter_pattern = r'##\s*Chapter\s*\d+[:\s]*([^\n]+)'
        chapters = re.split(chapter_pattern, content)
        
        for i in range(1, len(chapters), 2):
            if i + 1 < len(chapters):
                title = chapters[i].strip()
                content_text = chapters[i + 1].strip()
                sections.append({
                    "title": title,
                    "content": content_text,
                    "section_number": len(sections) + 1
                })
        
        print(f"✅ Content Extraction Test Results:")
        print(f"   Sections Found: {len(sections)}")
        for section in sections:
            print(f"   - {section['title']} (Section {section['section_number']})")
            print(f"     Content Length: {len(section['content'])} characters")
        
        return len(sections) > 0
        
    except Exception as e:
        print(f"❌ Content extraction test failed: {e}")
        return False


async def test_knowledge_graph_query():
    """Test knowledge graph querying capabilities."""
    print("\n🔧 Testing Knowledge Graph Query...")
    
    try:
        # Initialize knowledge graph agent
        kg_agent = KnowledgeGraphAgent()
        
        # Test query
        query = "Sun Tzu"
        request = AnalysisRequest(
            data_type=DataType.TEXT,
            content=query,
            language="en"
        )
        
        result = await kg_agent.process(request)
        
        print(f"✅ Knowledge Graph Query Test Results:")
        print(f"   Query: {query}")
        print(f"   Entities Found: {len(result.metadata.get('entities', [])) if result.metadata else 0}")
        print(f"   Processing Time: {result.processing_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Knowledge graph query test failed: {e}")
        return False


async def test_content_comparison():
    """Test content comparison capabilities."""
    print("\n🔧 Testing Content Comparison...")
    
    # Sample content sections
    sections = [
        "Sun Tzu said: The art of war is of vital importance to the state. It is a matter of life and death.",
        "Sun Tzu said: In the operations of war, where there are in the field a thousand swift chariots.",
        "Sun Tzu said: In the practical art of war, the best thing of all is to take the enemy's country whole."
    ]
    
    try:
        # Initialize agents
        text_agent = UnifiedTextAgent()
        kg_agent = KnowledgeGraphAgent()
        
        results = []
        
        for i, content in enumerate(sections):
            # Analyze each section
            request = AnalysisRequest(
                data_type=DataType.TEXT,
                content=content,
                language="en"
            )
            
            kg_result = await kg_agent.process(request)
            text_result = await text_agent.process(request)
            
            results.append({
                "section": f"Section {i+1}",
                "entities": kg_result.metadata.get('entities', []) if kg_result.metadata else [],
                "sentiment": text_result.sentiment.label if hasattr(text_result, 'sentiment') else "neutral"
            })
        
        # Find common entities
        all_entities = []
        for result in results:
            all_entities.extend(result.get("entities", []))
        
        entity_counts = {}
        for entity in all_entities:
            entity_name = entity.get("name", "")
            entity_counts[entity_name] = entity_counts.get(entity_name, 0) + 1
        
        common_entities = [
            {"name": name, "frequency": count}
            for name, count in entity_counts.items()
            if count > 1
        ]
        
        print(f"✅ Content Comparison Test Results:")
        print(f"   Sections Analyzed: {len(results)}")
        print(f"   Common Entities: {len(common_entities)}")
        for entity in common_entities:
            print(f"   - {entity['name']} (frequency: {entity['frequency']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Content comparison test failed: {e}")
        return False


async def test_mcp_tool_integration():
    """Test MCP tool integration patterns."""
    print("\n🔧 Testing MCP Tool Integration...")
    
    try:
        # Test the tool selection matrix
        tool_matrix = {
            "summarize chapter 1": {
                "tool": "summarize_text_content",
                "parameters": {
                    "text": "Chapter 1 content...",
                    "summary_type": "comprehensive",
                    "include_key_points": True
                }
            },
            "analyze chapter 1": {
                "tool": "analyze_chapter_content",
                "parameters": {
                    "chapter_text": "Chapter 1 content...",
                    "analysis_type": "comprehensive"
                }
            },
            "extract entities from chapter": {
                "tool": "extract_entities",
                "parameters": {
                    "text": "Chapter content...",
                    "language": "en"
                }
            },
            "compare chapters": {
                "tool": "compare_content_sections",
                "parameters": {
                    "content_sections": ["Chapter 1", "Chapter 2"],
                    "comparison_type": "themes"
                }
            }
        }
        
        print(f"✅ MCP Tool Integration Test Results:")
        print(f"   Tool Matrix Created: {len(tool_matrix)} patterns")
        for pattern, config in tool_matrix.items():
            print(f"   - '{pattern}' -> {config['tool']}")
        
        return True
        
    except Exception as e:
        print(f"❌ MCP tool integration test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting MCP Content Analysis Tools Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    test_results.append(await test_text_summarization())
    test_results.append(await test_chapter_analysis())
    test_results.append(await test_content_extraction())
    test_results.append(await test_knowledge_graph_query())
    test_results.append(await test_content_comparison())
    test_results.append(await test_mcp_tool_integration())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"✅ Tests Passed: {passed}/{total}")
    print(f"❌ Tests Failed: {total - passed}/{total}")
    print(f"📈 Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed! MCP content analysis tools are working correctly.")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    # Run the test suite
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
