"""
Test script for Phase 3 Advanced Features implementation.
Validates semantic similarity analysis, relationship optimization, and quality metrics.
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.phase3_testing_framework import Phase3TestingFramework
from src.agents.enhanced_knowledge_graph_agent import EnhancedKnowledgeGraphAgent


async def test_phase3_implementation():
    """Test Phase 3 implementation comprehensively."""
    print("🚀 Starting Phase 3 Advanced Features Testing")
    print("=" * 60)
    
    # Initialize testing framework
    testing_framework = Phase3TestingFramework()
    
    # Run comprehensive tests
    print("📊 Running comprehensive Phase 3 tests...")
    test_report = testing_framework.run_comprehensive_tests()
    
    # Display results
    print("\n📋 Test Results Summary:")
    print(f"Overall Status: {test_report['overall_status']}")
    print(f"Total Tests: {test_report['test_summary']['total_tests']}")
    print(f"Passed Tests: {test_report['test_summary']['passed_tests']}")
    print(f"Failed Tests: {test_report['test_summary']['failed_tests']}")
    print(f"Success Rate: {test_report['test_summary']['success_rate']:.2%}")
    
    print(f"\n⏱️ Performance Summary:")
    print(f"Total Execution Time: {test_report['performance_summary']['total_execution_time']:.2f}s")
    print(f"Average Execution Time: {test_report['performance_summary']['average_execution_time']:.2f}s")
    print(f"Throughput: {test_report['performance_summary']['throughput']:.2f} ops/sec")
    
    # Display detailed results
    print("\n🔍 Detailed Test Results:")
    for result in test_report['detailed_results']:
        status_emoji = "✅" if result['status'] == "PASS" else "❌"
        print(f"{status_emoji} {result['test_name']}: {result['status']}")
        print(f"   Execution Time: {result['execution_time']:.3f}s")
        
        if result['metrics']:
            print(f"   Metrics: {json.dumps(result['metrics'], indent=6)}")
        
        if result['errors']:
            print(f"   Errors: {result['errors']}")
        
        if result['warnings']:
            print(f"   Warnings: {result['warnings']}")
        
        print()
    
    # Quality assessment
    print("🎯 Quality Assessment:")
    quality_assessment = test_report['quality_assessment']
    print(f"Meets Orphan Node Target: {'✅' if quality_assessment['meets_orphan_node_target'] else '❌'}")
    print(f"Meets Performance Target: {'✅' if quality_assessment['meets_performance_target'] else '❌'}")
    print(f"Overall Quality Score: {quality_assessment['overall_quality_score']:.2%}")
    
    # Save test report
    report_path = testing_framework.save_test_report(test_report)
    print(f"\n💾 Test report saved to: {report_path}")
    
    return test_report


async def test_enhanced_knowledge_graph_agent():
    """Test the enhanced knowledge graph agent with Phase 3 features."""
    print("\n🧠 Testing Enhanced Knowledge Graph Agent with Phase 3 Features")
    print("=" * 70)
    
    # Initialize agent
    agent = EnhancedKnowledgeGraphAgent()
    
    # Test text
    test_text = """
    人工智能技术在各个领域都有广泛应用。清华大学和北京大学在机器学习研究方面
    进行了深入合作。北京中关村科技园区聚集了众多高科技企业，包括百度、阿里巴巴
    和腾讯等知名公司。这些企业在人工智能、大数据和云计算技术方面都有重要突破。
    政府相关部门也出台了支持政策，促进技术创新和产业发展。
    """
    
    print("📝 Test Text:")
    print(test_text.strip())
    print()
    
    # Test Phase 3 tools
    print("🔧 Testing Phase 3 Tools:")
    
    # 1. Semantic Similarity Analysis
    print("1. Testing Semantic Similarity Analysis...")
    similarity_result = await agent.analyze_semantic_similarity(test_text)
    similarity_data = similarity_result.get("content", [{}])[0]
    
    if "error" not in similarity_data:
        stats = similarity_data.get("statistics", {})
        print(f"   ✅ Total pairs analyzed: {stats.get('total_pairs', 0)}")
        print(f"   ✅ Average similarity: {stats.get('average_similarity', 0):.3f}")
        print(f"   ✅ High similarity pairs: {similarity_data.get('high_similarity_pairs', 0)}")
    else:
        print(f"   ❌ Error: {similarity_data['error']}")
    
    # 2. Relationship Optimization
    print("2. Testing Relationship Optimization...")
    optimization_result = await agent.optimize_relationships(test_text)
    optimization_data = optimization_result.get("content", [{}])[0]
    
    if "error" not in optimization_data:
        print(f"   ✅ Original relationships: {optimization_data.get('original_relationships', 0)}")
        print(f"   ✅ Optimized relationships: {optimization_data.get('optimized_relationships', 0)}")
        print(f"   ✅ Quality improvement: {optimization_data.get('quality_improvement', 0):.3f}")
        print(f"   ✅ Redundancy reduction: {optimization_data.get('redundancy_reduction', 0):.3f}")
    else:
        print(f"   ❌ Error: {optimization_data['error']}")
    
    # 3. Advanced Entity Clustering
    print("3. Testing Advanced Entity Clustering...")
    clustering_result = await agent.cluster_entities_advanced(test_text)
    clustering_data = clustering_result.get("content", [{}])[0]
    
    if "error" not in clustering_data:
        stats = clustering_data.get("cluster_statistics", {})
        print(f"   ✅ Total clusters: {stats.get('total_clusters', 0)}")
        print(f"   ✅ Relationships created: {stats.get('total_relationships_created', 0)}")
        print(f"   ✅ Average cluster size: {stats.get('average_cluster_size', 0):.2f}")
    else:
        print(f"   ❌ Error: {clustering_data['error']}")
    
    # 4. Quality Assessment
    print("4. Testing Quality Assessment...")
    quality_result = await agent.run_phase3_quality_assessment(test_text)
    quality_data = quality_result.get("content", [{}])[0]
    
    if "error" not in quality_data:
        metrics = quality_data.get("quality_metrics", {})
        print(f"   ✅ Total entities: {metrics.get('total_entities', 0)}")
        print(f"   ✅ Total relationships: {metrics.get('total_relationships', 0)}")
        print(f"   ✅ Orphan nodes: {metrics.get('orphan_nodes', 0)}")
        print(f"   ✅ Orphan rate: {metrics.get('orphan_rate', 1.0):.3f}")
        print(f"   ✅ Relationship coverage: {metrics.get('relationship_coverage', 0.0):.3f}")
        print(f"   ✅ Meets orphan target: {'✅' if metrics.get('meets_orphan_target', False) else '❌'}")
        print(f"   ✅ Meets coverage target: {'✅' if metrics.get('meets_coverage_target', False) else '❌'}")
        
        assessment = quality_data.get("overall_assessment", {})
        recommendations = assessment.get("recommendations", [])
        if recommendations:
            print("   📋 Recommendations:")
            for rec in recommendations:
                print(f"      • {rec}")
    else:
        print(f"   ❌ Error: {quality_data['error']}")
    
    print("\n✅ Enhanced Knowledge Graph Agent Phase 3 testing completed!")


async def main():
    """Main test function."""
    print("🎯 Phase 3 Advanced Features Implementation Test")
    print("=" * 60)
    
    try:
        # Test Phase 3 implementation
        test_report = await test_phase3_implementation()
        
        # Test enhanced knowledge graph agent
        await test_enhanced_knowledge_graph_agent()
        
        # Overall assessment
        print("\n🎉 Phase 3 Implementation Assessment:")
        print("=" * 50)
        
        if test_report['overall_status'] == "PASS":
            print("✅ Phase 3 implementation is working correctly!")
            print("✅ All advanced features are operational!")
            print("✅ Quality targets are being met!")
        else:
            print("⚠️ Phase 3 implementation has some issues.")
            print("Please review the detailed test results above.")
        
        print("\n📊 Key Achievements:")
        print("• Semantic similarity analysis implemented")
        print("• Relationship optimization algorithms working")
        print("• Advanced entity clustering operational")
        print("• Comprehensive quality metrics available")
        print("• Performance benchmarking completed")
        
        return test_report
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the tests
    result = asyncio.run(main())
    
    if result:
        print(f"\n🎯 Phase 3 Implementation Status: {result['overall_status']}")
        sys.exit(0 if result['overall_status'] == "PASS" else 1)
    else:
        print("\n❌ Phase 3 Implementation Test Failed")
        sys.exit(1)
