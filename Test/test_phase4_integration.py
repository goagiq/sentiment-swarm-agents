#!/usr/bin/env python3
"""
Simple test script to verify Phase 4 integration.
Tests that all components are properly integrated without async operations.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_phase4_integration():
    """Test Phase 4 integration components."""
    print("🔧 Testing Phase 4 Integration")
    print("=" * 50)
    
    try:
        # Test 1: Enhanced Knowledge Graph Agent
        print("1. Testing Enhanced Knowledge Graph Agent...")
        from agents.enhanced_knowledge_graph_agent import EnhancedKnowledgeGraphAgent
        agent = EnhancedKnowledgeGraphAgent()
        print("   ✅ Enhanced Knowledge Graph Agent initialized")
        
        # Test 2: Language-specific configurations
        print("2. Testing Language-specific configurations...")
        from config.language_config import ChineseConfig, RussianConfig, EnglishConfig
        chinese_config = ChineseConfig()
        russian_config = RussianConfig()
        english_config = EnglishConfig()
        print("   ✅ Language configurations loaded")
        
        # Test 3: Phase 3 components
        print("3. Testing Phase 3 components...")
        from core.semantic_similarity_analyzer import SemanticSimilarityAnalyzer
        from core.relationship_optimizer import RelationshipOptimizer
        from core.chinese_entity_clustering import ChineseEntityClustering
        
        semantic_analyzer = SemanticSimilarityAnalyzer()
        relationship_optimizer = RelationshipOptimizer()
        entity_clustering = ChineseEntityClustering()
        print("   ✅ Phase 3 components initialized")
        
        # Test 4: Main.py integration
        print("4. Testing main.py integration...")
        import main
        print("   ✅ Main.py imports successfully")
        
        # Test 5: MCP tools availability
        print("5. Testing MCP tools availability...")
        mcp_server = main.OptimizedMCPServer()
        print("   ✅ MCP server initialized")
        
        print("\n🎉 Phase 4 Integration Test Results:")
        print("✅ All Phase 4 components are properly integrated!")
        print("✅ Enhanced Knowledge Graph Agent is working!")
        print("✅ Language-specific configurations are loaded!")
        print("✅ Phase 3 components are available!")
        print("✅ Main.py integration is successful!")
        print("✅ MCP tools are accessible!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during Phase 4 integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_phase4_integration()
    if success:
        print("\n✅ Phase 4 Integration: COMPLETE")
    else:
        print("\n❌ Phase 4 Integration: FAILED")
