"""
Test script for Knowledge Graph Integration.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.knowledge_graph_integration import KnowledgeGraphIntegration


async def test_knowledge_graph_integration():
    """Test the knowledge graph integration functionality."""
    
    print("🧪 Testing Knowledge Graph Integration")
    print("=" * 60)
    
    try:
        # Initialize the integration
        integration = KnowledgeGraphIntegration()
        print("✅ Integration initialized successfully")
        
        # Sample article content from Russian-Ukraine war articles
        sample_articles = [
            """
            German Chancellor Olaf Scholz expressed hope for a breakthrough in Ukraine talks during the upcoming meeting between Russian President Vladimir Putin and US President Donald Trump on Alaska. Scholz stated that Germany is willing to participate in talks aimed at resolving the conflict in Ukraine.
            """,
            """
            Republican vice-presidential candidate James David Vance stated that Donald Trump wants to end funding for the Ukraine conflict. Vance emphasized that Trump believes the US should focus on solving domestic problems rather than financing foreign conflicts.
            """,
            """
            Russian air defense forces destroyed Ukrainian unmanned aerial vehicles in two regions of the Chechen Republic. The Russian Ministry of Defense reported that all attempts by Ukrainian forces to attack Russian objects were successfully repelled.
            """,
            """
            Ministers of Foreign Affairs of the European Union's member states are scheduled to meet on Monday for an emergency meeting to discuss the situation in Ukraine. The European Commission announced that the meeting will discuss measures to support Ukraine and sanctions against Russia.
            """,
            """
            A White House representative stated that the US administration does not rule out the participation of Ukrainian President Volodymyr Zelensky in the upcoming summit between Russian President Vladimir Putin and US President Donald Trump "in some form." The format of the Ukrainian leader's participation is still being discussed.
            """
        ]
        
        print(f"📝 Processing {len(sample_articles)} articles with integration...")
        
        # Process with integration
        results = await integration.process_with_improved_extraction(sample_articles)
        
        print(f"✅ Integration processing completed!")
        
        # Show improved utility results
        improved_results = results['improved_utility_results']
        print(f"\n📊 Improved Utility Results:")
        print(f"   Entities extracted: {improved_results['entities_extracted']}")
        print(f"   Relationships mapped: {improved_results['relationships_mapped']}")
        print(f"   Graph nodes: {improved_results['graph_nodes']}")
        print(f"   Graph edges: {improved_results['graph_edges']}")
        
        # Show original agent results
        agent_results = results['agent_results']
        print(f"\n📊 Original Agent Results:")
        print(f"   Entities extracted: {agent_results['entities_extracted']}")
        print(f"   Relationships mapped: {agent_results['relationships_mapped']}")
        print(f"   Graph nodes: {agent_results['graph_nodes']}")
        print(f"   Graph edges: {agent_results['graph_edges']}")
        
        # Show comparison
        comparison = results['comparison']
        print(f"\n📈 Performance Comparison:")
        print(f"   Entity improvement: {comparison['improvement']['entities_improvement']}")
        print(f"   Relationship improvement: {comparison['improvement']['relationships_improvement']}")
        print(f"   Node improvement: {comparison['improvement']['nodes_improvement']}")
        print(f"   Edge improvement: {comparison['improvement']['edges_improvement']}")
        
        # Show file paths
        print(f"\n📁 Generated Files:")
        print(f"   Integration report: {results['integration_report']}")
        print(f"   Improved utility summary: {improved_results['summary_report']}")
        print(f"   PNG visualization: {improved_results['visualization_results']['png_file']}")
        print(f"   HTML visualization: {improved_results['visualization_results']['html_file']}")
        
        # Validate integration
        print(f"\n🔍 Validating integration...")
        validation = await integration.validate_integration(sample_articles)
        
        print(f"   Integration validation: {'✅ Success' if validation['integration_successful'] else '❌ Failed'}")
        print(f"   Files created: {len(validation['files_created'])}")
        
        if validation['validation_errors']:
            print(f"   Validation errors: {len(validation['validation_errors'])}")
            for error in validation['validation_errors']:
                print(f"     - {error}")
        
        print(f"\n✅ All integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_knowledge_graph_integration())
    sys.exit(0 if success else 1)
