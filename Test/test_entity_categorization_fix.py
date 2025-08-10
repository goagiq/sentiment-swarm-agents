#!/usr/bin/env python3
"""
Test script to verify entity categorization fix and report generation.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.knowledge_graph_agent import KnowledgeGraphAgent
from config.settings import settings


async def test_entity_categorization():
    """Test that entities are properly categorized and not all CONCEPT."""
    print("🧪 Testing Entity Categorization Fix...")
    
    # Initialize knowledge graph agent
    agent = KnowledgeGraphAgent()
    
    # Test text with various entity types
    test_text = """
    Donald Trump announced new tariffs on China and Mexico. The European Union 
    criticized these policies. President Biden responded from Washington DC. 
    The White House administration implemented trade restrictions. 
    Companies like Intel and Microsoft are affected by these tariffs.
    """
    
    # Extract entities
    result = await agent.extract_entities(test_text)
    
    print(f"📊 Entity extraction result: {result}")
    
    if 'content' in result and len(result['content']) > 0:
        content = result['content'][0].get('json', {})
        entities = content.get('entities', [])
        print(f"\n📋 Found {len(entities)} entities:")
        
        # Check entity types
        entity_types = {}
        for entity in entities:
            entity_type = entity.get('type', 'UNKNOWN')
            entity_name = entity.get('name', 'Unknown')
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            print(f"  - {entity_name} ({entity_type})")
        
        print(f"\n📈 Entity type distribution:")
        for entity_type, count in entity_types.items():
            print(f"  - {entity_type}: {count}")
        
        # Check if all entities are not CONCEPT
        concept_count = entity_types.get('CONCEPT', 0)
        total_count = len(entities)
        
        if concept_count < total_count:
            print(f"✅ SUCCESS: Only {concept_count}/{total_count} entities are CONCEPT")
            return True
        else:
            print(f"❌ FAILURE: All {total_count} entities are CONCEPT")
            return False
    else:
        print("❌ FAILURE: No entities found in result")
        return False


async def test_report_generation():
    """Test that reports are generated in the Results directory with proper titles."""
    print("\n📄 Testing Report Generation...")
    
    # Initialize knowledge graph agent
    agent = KnowledgeGraphAgent()
    
    # Add some test data to the graph
    test_entities = [
        {"name": "Donald Trump", "type": "PERSON", "confidence": 0.9},
        {"name": "China", "type": "LOCATION", "confidence": 0.8},
        {"name": "European Union", "type": "ORGANIZATION", "confidence": 0.8},
        {"name": "Tariffs", "type": "CONCEPT", "confidence": 0.7},
        {"name": "Trade Policy", "type": "CONCEPT", "confidence": 0.7}
    ]
    
    test_relationships = [
        {"source": "Donald Trump", "target": "China", "relationship_type": "IMPLEMENTS", "confidence": 0.8},
        {"source": "Donald Trump", "target": "Tariffs", "relationship_type": "CREATES", "confidence": 0.8},
        {"source": "European Union", "target": "Tariffs", "relationship_type": "OPPOSES", "confidence": 0.7}
    ]
    
    # Add to graph
    await agent._add_to_graph(test_entities, test_relationships, "test_request")
    
    # Generate report
    result = await agent.generate_graph_report()
    
    print(f"📊 Report generation result: {result}")
    
    if 'content' in result and len(result['content']) > 0:
        report_data = result['content'][0].get('json', {})
        
        # Check if files were generated
        png_file = report_data.get('png_file')
        html_file = report_data.get('html_file')
        md_file = report_data.get('md_file')
        
        print(f"\n📁 Generated files:")
        if png_file:
            print(f"  - PNG: {png_file}")
        if html_file:
            print(f"  - HTML: {html_file}")
        if md_file:
            print(f"  - Markdown: {md_file}")
        
        # Check if files are in Results directory
        results_dir = settings.paths.results_dir
        files_in_results = []
        
        for file_path in [png_file, html_file, md_file]:
            if file_path and Path(file_path).parent.name == "reports":
                files_in_results.append(file_path)
        
        if files_in_results:
            print(f"✅ SUCCESS: {len(files_in_results)} files generated in Results directory")
            
            # Check file contents for proper titles
            for file_path in files_in_results:
                if file_path and Path(file_path).exists():
                    if file_path.endswith('.md'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if settings.report_generation.report_title_prefix in content:
                                print(f"✅ SUCCESS: Markdown file has proper title")
                            else:
                                print(f"❌ FAILURE: Markdown file missing proper title")
            
            return True
        else:
            print(f"❌ FAILURE: No files generated in Results directory")
            return False
    else:
        print("❌ FAILURE: Report generation failed")
        return False


async def test_settings_integration():
    """Test that settings are properly integrated."""
    print("\n⚙️ Testing Settings Integration...")
    
    # Check if settings are accessible
    try:
        print(f"📁 Results directory: {settings.paths.results_dir}")
        print(f"📁 Reports directory: {settings.paths.reports_dir}")
        print(f"📁 Knowledge graphs directory: {settings.paths.knowledge_graphs_dir}")
        print(f"📝 Report title prefix: {settings.report_generation.report_title_prefix}")
        print(f"📝 Report filename prefix: {settings.report_generation.report_filename_prefix}")
        
        # Check entity types
        entity_types = settings.entity_categorization.entity_types
        print(f"🏷️ Entity types configured: {list(entity_types.keys())}")
        
        print("✅ SUCCESS: Settings integration working")
        return True
        
    except Exception as e:
        print(f"❌ FAILURE: Settings integration failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting Entity Categorization and Report Generation Tests\n")
    
    # Run tests
    tests = [
        test_settings_integration,
        test_entity_categorization,
        test_report_generation
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*50)
    print("📋 TEST SUMMARY")
    print("="*50)
    
    test_names = [
        "Settings Integration",
        "Entity Categorization", 
        "Report Generation"
    ]
    
    for i, (test_name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{i+1}. {test_name}: {status}")
    
    all_passed = all(results)
    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
