#!/usr/bin/env python3
"""
Multi-Domain Knowledge Graph System Demo

This demo showcases the new multi-domain knowledge graph system that solves
the "one pot" problem by implementing language-based content isolation.

Features demonstrated:
1. Language-based domain separation
2. Cross-domain relationship detection
3. Topic categorization
4. Flexible querying patterns
5. Multiple visualization modes
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.agents.multi_domain_knowledge_graph_agent import MultiDomainKnowledgeGraphAgent
from src.agents.multi_domain_visualization_agent import MultiDomainVisualizationAgent
from src.core.models import AnalysisRequest, DataType


class MultiDomainKnowledgeGraphDemo:
    """Demo class for the Multi-Domain Knowledge Graph System."""
    
    def __init__(self):
        self.knowledge_agent = MultiDomainKnowledgeGraphAgent()
        self.viz_agent = MultiDomainVisualizationAgent()
        
        # Sample content for demonstration
        self.sample_content = {
            "english_politics": {
                "text": """
                President Trump announced new tariffs on Chinese imports today. 
                The trade war between the United States and China continues to escalate.
                The White House press secretary confirmed the new policy.
                """,
                "expected_language": "en",
                "expected_topics": ["politics", "economics"]
            },
            "chinese_economics": {
                "text": """
                特朗普总统今天宣布对中国进口商品征收新关税。
                中美贸易战继续升级。
                白宫新闻秘书确认了新政策。
                """,
                "expected_language": "zh",
                "expected_topics": ["politics", "economics"]
            },
            "spanish_social": {
                "text": """
                El presidente Trump anunció nuevos aranceles a las importaciones chinas hoy.
                La guerra comercial entre Estados Unidos y China continúa escalando.
                El secretario de prensa de la Casa Blanca confirmó la nueva política.
                """,
                "expected_language": "es",
                "expected_topics": ["politics", "economics"]
            },
            "french_tech": {
                "text": """
                Apple a annoncé de nouveaux produits technologiques aujourd'hui.
                L'iPhone 15 Pro Max présente des fonctionnalités révolutionnaires.
                La technologie de reconnaissance faciale s'est considérablement améliorée.
                """,
                "expected_language": "fr",
                "expected_topics": ["tech", "science"]
            },
            "german_science": {
                "text": """
                Wissenschaftler haben einen Durchbruch in der Quantencomputer-Forschung erzielt.
                Die neue Technologie könnte die Kryptographie revolutionieren.
                Forschungsinstitute weltweit arbeiten an ähnlichen Projekten.
                """,
                "expected_language": "de",
                "expected_topics": ["science", "tech"]
            }
        }
    
    async def run_demo(self):
        """Run the complete demo."""
        print("🚀 Multi-Domain Knowledge Graph System Demo")
        print("=" * 50)
        
        # Step 1: Process sample content
        await self.demo_content_processing()
        
        # Step 2: Demonstrate querying capabilities
        await self.demo_querying()
        
        # Step 3: Show cross-domain relationships
        await self.demo_cross_domain_analysis()
        
        # Step 4: Generate reports
        await self.demo_reporting()
        
        # Step 5: Demonstrate visualizations
        await self.demo_visualizations()
        
        print("\n✅ Demo completed successfully!")
    
    async def demo_content_processing(self):
        """Demonstrate content processing with language detection."""
        print("\n📝 Step 1: Content Processing")
        print("-" * 30)
        
        for content_name, content_data in self.sample_content.items():
            print(f"\nProcessing: {content_name}")
            
            # Create analysis request
            request = AnalysisRequest(
                data_type=DataType.TEXT,
                content=content_data["text"],
                request_id=f"demo_{content_name}"
            )
            
            # Process content
            result = await self.knowledge_agent.process(request)
            
            # Display results
            detected_lang = result.metadata.get("detected_language", "unknown")
            detected_topics = result.metadata.get("detected_topics", [])
            entities_count = result.metadata.get("entities_extracted", 0)
            relationships_count = result.metadata.get("relationships_mapped", 0)
            
            print(f"  Expected Language: {content_data['expected_language']}")
            print(f"  Detected Language: {detected_lang}")
            print(f"  Expected Topics: {content_data['expected_topics']}")
            print(f"  Detected Topics: {detected_topics}")
            print(f"  Entities Extracted: {entities_count}")
            print(f"  Relationships Mapped: {relationships_count}")
            
            # Verify language detection
            if detected_lang == content_data["expected_language"]:
                print("  ✅ Language detection: CORRECT")
            else:
                print(f"  ⚠️  Language detection: EXPECTED {content_data['expected_language']}, GOT {detected_lang}")
    
    async def demo_querying(self):
        """Demonstrate different querying patterns."""
        print("\n🔍 Step 2: Querying Capabilities")
        print("-" * 30)
        
        # Query within specific domains
        print("\nDomain-specific queries:")
        for lang in ["en", "zh", "es", "fr", "de"]:
            results = await self.knowledge_agent.query_domain("Trump", lang)
            if results["status"] == "success" and results["count"] > 0:
                print(f"  {lang.upper()}: Found {results['count']} entities")
            else:
                print(f"  {lang.upper()}: No results")
        
        # Cross-domain query
        print("\nCross-domain query for 'Trump':")
        cross_results = await self.knowledge_agent.query_cross_domain("Trump")
        if cross_results["status"] == "success":
            print(f"  Found in {len(cross_results['results'])} domains")
            for domain, domain_results in cross_results["results"].items():
                print(f"    {domain.upper()}: {domain_results['count']} entities")
        
        # Comprehensive query
        print("\nComprehensive query for 'technology':")
        all_results = await self.knowledge_agent.query_all_domains("technology")
        if all_results["status"] == "success":
            print(f"  Found {all_results['count']} entities across all domains")
    
    async def demo_cross_domain_analysis(self):
        """Demonstrate cross-domain relationship analysis."""
        print("\n🌐 Step 3: Cross-Domain Analysis")
        print("-" * 30)
        
        # Analyze cross-domain connections
        cross_connections = await self.knowledge_agent.analyze_cross_domain_connections()
        if cross_connections["status"] == "success":
            print(f"Cross-domain connections found: {cross_connections['count']}")
            
            if cross_connections["connections"]:
                print("Sample connections:")
                for i, connection in enumerate(cross_connections["connections"][:3]):
                    print(f"  {i+1}. {connection['source']} -> {connection['target']}")
        
        # Find entity paths across domains
        print("\nFinding paths between entities:")
        paths = await self.knowledge_agent.find_entity_paths_multi_domain("Trump", "technology", max_paths=3)
        if paths["status"] == "success":
            print(f"Found {paths['count']} paths between 'Trump' and 'technology'")
    
    async def demo_reporting(self):
        """Demonstrate report generation."""
        print("\n📊 Step 4: Report Generation")
        print("-" * 30)
        
        # Get comprehensive statistics
        stats = await self.knowledge_agent.get_domain_statistics()
        if stats["status"] == "success":
            print("Overall Statistics:")
            print(f"  Total Domains: {stats['statistics']['total_domains']}")
            print(f"  Total Nodes: {stats['statistics']['total_nodes']}")
            print(f"  Total Edges: {stats['statistics']['total_edges']}")
            
            print("\nDomain-specific statistics:")
            for lang, domain_stats in stats["statistics"]["domains"].items():
                print(f"  {lang.upper()}: {domain_stats['nodes']} nodes, {domain_stats['edges']} edges")
        
        # Generate domain reports
        print("\nGenerating domain reports:")
        for lang in ["en", "zh", "es", "fr", "de"]:
            report = await self.knowledge_agent.generate_domain_report(lang)
            if report["status"] == "success":
                report_data = report["report"]
                print(f"  {lang.upper()}: {report_data['statistics']['nodes']} nodes, "
                      f"{report_data['statistics']['edges']} edges")
        
        # Generate comprehensive report
        print("\nGenerating comprehensive report:")
        comprehensive_report = await self.knowledge_agent.generate_comprehensive_report()
        if comprehensive_report["status"] == "success":
            print("  ✅ Comprehensive report generated successfully")
    
    async def demo_visualizations(self):
        """Demonstrate visualization capabilities."""
        print("\n🎨 Step 5: Visualization Capabilities")
        print("-" * 30)
        
        # Prepare sample graph data for visualization
        graphs_data = {}
        for lang in ["en", "zh", "es", "fr", "de"]:
            # Get domain statistics
            stats = await self.knowledge_agent.get_domain_statistics()
            if stats["status"] == "success" and lang in stats["statistics"]["domains"]:
                domain_stats = stats["statistics"]["domains"][lang]
                
                # Create sample visualization data
                graphs_data[lang] = {
                    "nodes": [
                        {
                            "name": f"Sample Entity {i}",
                            "type": "person" if i % 2 == 0 else "organization",
                            "topics": ["politics", "economics"] if i % 2 == 0 else ["tech"],
                            "x": i * 10,
                            "y": i * 5,
                            "size": 10 + i
                        }
                        for i in range(1, 6)
                    ],
                    "edges": [
                        {
                            "source": f"Sample Entity {i}",
                            "target": f"Sample Entity {i+1}",
                            "relationship_type": "related",
                            "source_x": i * 10,
                            "source_y": i * 5,
                            "target_x": (i+1) * 10,
                            "target_y": (i+1) * 5
                        }
                        for i in range(1, 5)
                    ]
                }
        
        # Create separate domain visualizations
        print("Creating separate domain visualizations:")
        separate_results = await self.viz_agent.visualize_separate_domains(
            graphs_data,
            options={"format": "html", "max_nodes": 50}
        )
        if separate_results["status"] == "success":
            print(f"  ✅ Created {separate_results['total_domains']} separate visualizations")
        
        # Create combined view
        print("Creating combined view visualization:")
        combined_results = await self.viz_agent.visualize_combined_view(
            graphs_data,
            options={
                "selected_domains": ["en", "zh", "es"],
                "selected_topics": ["politics", "economics"],
                "max_nodes": 100
            }
        )
        if combined_results["status"] == "success":
            print(f"  ✅ Combined view created with {combined_results['node_count']} nodes")
        
        # Create hierarchical view
        print("Creating hierarchical view visualization:")
        hierarchical_results = await self.viz_agent.visualize_hierarchical_view(graphs_data)
        if hierarchical_results["status"] == "success":
            print(f"  ✅ Hierarchical view created with {hierarchical_results['domain_count']} domains")
        
        # Create interactive dashboard
        print("Creating interactive dashboard:")
        dashboard_results = await self.viz_agent.create_interactive_dashboard(graphs_data)
        if dashboard_results["status"] == "success":
            print("  ✅ Interactive dashboard created")
        
        # Create topic analysis chart
        print("Creating topic analysis chart:")
        topic_results = await self.viz_agent.create_topic_analysis_chart(graphs_data)
        if topic_results["status"] == "success":
            print(f"  ✅ Topic analysis chart created with {topic_results['topic_count']} topics")
    
    async def demo_advanced_features(self):
        """Demonstrate advanced features."""
        print("\n🚀 Step 6: Advanced Features")
        print("-" * 30)
        
        # Analyze communities within domains
        print("Analyzing communities within domains:")
        for lang in ["en", "zh", "es"]:
            communities = await self.knowledge_agent.analyze_domain_communities(lang)
            if communities["status"] == "success":
                print(f"  {lang.upper()}: Found {communities['count']} communities")
        
        # Get entity context across domains
        print("\nGetting entity context across domains:")
        context = await self.knowledge_agent.get_entity_context_multi_domain("Trump")
        if context["status"] == "success":
            appearances = context["context"]["appearances"]
            print(f"  Entity 'Trump' appears in {len(appearances)} domains")
            for domain, data in appearances.items():
                print(f"    {domain.upper()}: {data['type']} with {len(data['neighbors'])} neighbors")
        
        # Merge related domains
        print("\nMerging related domains:")
        merge_results = await self.knowledge_agent.merge_related_domains("en", "zh")
        if merge_results["status"] == "success":
            print(f"  Found {merge_results['count']} related entities between English and Chinese")


async def main():
    """Main demo function."""
    demo = MultiDomainKnowledgeGraphDemo()
    
    try:
        await demo.run_demo()
        
        # Optional: Run advanced features demo
        print("\n" + "=" * 50)
        print("Advanced Features Demo")
        print("=" * 50)
        await demo.demo_advanced_features()
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
