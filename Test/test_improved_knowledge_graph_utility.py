"""
Test script for Improved Knowledge Graph Utility.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.improved_knowledge_graph_utility import ImprovedKnowledgeGraphUtility


async def test_improved_knowledge_graph_utility():
    """Test the improved knowledge graph utility functionality."""
    
    print("🧪 Testing Improved Knowledge Graph Utility")
    print("=" * 60)
    
    try:
        # Initialize the utility
        utility = ImprovedKnowledgeGraphUtility()
        print("✅ Utility initialized successfully")
        
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
            """,
            """
            Representatives of the Ukrainian Orthodox Church (UOC) accused the Ukrainian Autocephalous Orthodox Church (UAC) of being pro-Russian after a UAC seminarian appeared in a photo wearing a shirt with a Russian inscription. The UOC claimed this was evidence that the UAC supports Russia's aggression against Ukraine.
            """,
            """
            Russian troops of the "South" grouping destroyed a 120mm mortar of the Ukrainian Armed Forces in the village of Novodmitrovka. The Russian Ministry of Defense reported that Russian forces continue to conduct successful operations to destroy Ukrainian military equipment and personnel.
            """,
            """
            General Alexander Lapin was replaced by General Eugene Nikiforov as commander of the Leningrad Military District. The change in military leadership comes as Russian forces continue operations in the Kursk region.
            """
        ]
        
        print(f"📝 Processing {len(sample_articles)} articles...")
        
        # Process articles and create graph
        results = await utility.process_articles_and_create_graph(sample_articles)
        
        print(f"✅ Processing completed!")
        print(f"   Entities extracted: {results['entities_extracted']}")
        print(f"   Relationships mapped: {results['relationships_mapped']}")
        print(f"   Graph nodes: {results['graph_nodes']}")
        print(f"   Graph edges: {results['graph_edges']}")
        
        # Show visualization results
        viz_results = results['visualization_results']
        print(f"\n📊 Visualization Results:")
        print(f"   PNG file: {viz_results['png_file']}")
        print(f"   HTML file: {viz_results['html_file']}")
        print(f"   Graph data file: {viz_results['graph_file']}")
        
        # Show validation results
        print(f"\n🔍 File Validation Results:")
        for file_path, validation in viz_results['validation'].items():
            status = "✅ Valid" if validation['valid'] else "❌ Invalid"
            print(f"   {Path(file_path).name}: {status} (Size: {validation['size']} bytes)")
            if validation['error']:
                print(f"     Error: {validation['error']}")
        
        # Show graph statistics
        stats = viz_results['graph_stats']
        print(f"\n📈 Graph Statistics:")
        print(f"   Density: {stats.get('density', 0):.4f}")
        print(f"   Average Clustering: {stats.get('average_clustering', 0):.4f}")
        print(f"   Connected Components: {stats.get('connected_components', 1)}")
        
        # Show summary report
        print(f"\n📋 Summary Report: {results['summary_report']}")
        
        print(f"\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_improved_knowledge_graph_utility())
    sys.exit(0 if success else 1)
