"""
Test script for Knowledge Graph Agent.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from agents.knowledge_graph_agent import KnowledgeGraphAgent
from core.models import AnalysisRequest, DataType


async def test_knowledge_graph_agent():
    """Test the knowledge graph agent functionality."""
    
    print("🧪 Testing Knowledge Graph Agent")
    print("=" * 50)
    
    try:
        # Initialize the agent
        agent = KnowledgeGraphAgent()
        print("✅ Agent initialized successfully")
        
        # Test with a simple text
        test_text = """
        John Smith is a software engineer at Google. He works with Sarah Johnson 
        on machine learning projects. Google is headquartered in Mountain View, 
        California. The company was founded by Larry Page and Sergey Brin.
        """
        
        # Create analysis request
        request = AnalysisRequest(
            data_type=DataType.TEXT,
            content=test_text,
            language="en"
        )
        
        print(f"📝 Processing test text...")
        result = await agent.process(request)
        
        print(f"✅ Processing completed!")
        print(f"   Status: {result.status}")
        print(f"   Entities extracted: {result.metadata.get('entities_extracted', 0)}")
        print(f"   Relationships mapped: {result.metadata.get('relationships_mapped', 0)}")
        print(f"   Graph nodes: {result.metadata.get('graph_nodes', 0)}")
        print(f"   Graph edges: {result.metadata.get('graph_edges', 0)}")
        
        # Test individual tools
        print(f"\n🔧 Testing individual tools...")
        
        # Test entity extraction
        entities_result = await agent.extract_entities(test_text)
        print(f"   Entity extraction: {len(entities_result.get('content', [{}])[0].get('json', {}).get('entities', []))} entities found")
        
        # Test graph querying
        query_result = await agent.query_knowledge_graph("Google")
        print(f"   Graph query: Query completed")
        
        # Test graph report generation
        report_result = await agent.generate_graph_report()
        print(f"   Graph report: {report_result.get('content', [{}])[0].get('json', {}).get('message', 'Unknown')}")
        
        # Show final statistics
        stats = agent._get_graph_stats()
        print(f"\n📊 Final Statistics:")
        print(f"   Total nodes: {stats['nodes']}")
        print(f"   Total edges: {stats['edges']}")
        print(f"   Graph density: {stats['density']:.4f}")
        
        print(f"\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_knowledge_graph_agent())
    sys.exit(0 if success else 1)
