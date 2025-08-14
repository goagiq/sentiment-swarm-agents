#!/usr/bin/env python3
"""
Test Enhanced Decision Support System

Tests the enhanced decision support system with knowledge graph integration,
multilingual support, and new MCP tools.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger
from src.core.unified_mcp_client import call_unified_mcp_tool
from src.config.decision_support_config import (
    get_decision_support_config,
    get_language_decision_config,
    get_language_entity_patterns
)


class EnhancedDecisionSupportTester:
    """Test suite for enhanced decision support system."""
    
    def __init__(self):
        self.config = get_decision_support_config()
        self.test_results = []
        self.start_time = datetime.now()
        
    async def run_all_tests(self):
        """Run all enhanced decision support tests."""
        logger.info("🚀 Starting Enhanced Decision Support System Tests")
        logger.info("=" * 60)
        
        # Test 1: Configuration Loading
        await self.test_configuration_loading()
        
        # Test 2: Knowledge Graph Integration
        await self.test_knowledge_graph_integration()
        
        # Test 3: Multilingual Support
        await self.test_multilingual_support()
        
        # Test 4: Decision Context Extraction
        await self.test_decision_context_extraction()
        
        # Test 5: Entity Extraction for Decisions
        await self.test_entity_extraction_for_decisions()
        
        # Test 6: Decision Pattern Analysis
        await self.test_decision_pattern_analysis()
        
        # Test 7: Recommendation Generation
        await self.test_recommendation_generation()
        
        # Test 8: Action Prioritization
        await self.test_action_prioritization()
        
        # Test 9: Implementation Planning
        await self.test_implementation_planning()
        
        # Test 10: Success Prediction
        await self.test_success_prediction()
        
        # Generate test report
        await self.generate_test_report()
    
    async def test_configuration_loading(self):
        """Test configuration loading and validation."""
        logger.info("📋 Test 1: Configuration Loading")
        
        try:
            # Test main configuration
            config = get_decision_support_config()
            assert config is not None, "Main configuration should not be None"
            assert hasattr(config, 'knowledge_graph'), "Should have knowledge_graph config"
            assert hasattr(config, 'multilingual'), "Should have multilingual config"
            assert hasattr(config, 'real_time_data'), "Should have real_time_data config"
            
            # Test language-specific configurations
            en_config = get_language_decision_config("en")
            zh_config = get_language_decision_config("zh")
            ja_config = get_language_decision_config("ja")
            
            assert en_config["decision_style"] == "analytical", "English should be analytical"
            assert zh_config["decision_style"] == "holistic", "Chinese should be holistic"
            assert ja_config["decision_style"] == "consensus_based", "Japanese should be consensus-based"
            
            # Test entity patterns
            en_patterns = get_language_entity_patterns("en")
            zh_patterns = get_language_entity_patterns("zh")
            
            assert "business_terms" in en_patterns, "English should have business terms"
            assert "business_terms" in zh_patterns, "Chinese should have business terms"
            
            self.test_results.append({
                "test": "Configuration Loading",
                "status": "PASSED",
                "details": "All configuration components loaded successfully"
            })
            logger.info("✅ Configuration Loading: PASSED")
            
        except Exception as e:
            self.test_results.append({
                "test": "Configuration Loading",
                "status": "FAILED",
                "error": str(e)
            })
            logger.error(f"❌ Configuration Loading: FAILED - {e}")
    
    async def test_knowledge_graph_integration(self):
        """Test knowledge graph integration capabilities."""
        logger.info("🔗 Test 2: Knowledge Graph Integration")
        
        try:
            # Test decision context query
            result = await call_unified_mcp_tool(
                "query_decision_context",
                {
                    "content": "Our company needs to improve operational efficiency and reduce costs while maintaining quality standards.",
                    "language": "en",
                    "context_type": "comprehensive"
                }
            )
            
            assert result["success"], f"Decision context query failed: {result.get('error', 'Unknown error')}"
            assert "result" in result, "Result should be present in response"
            
            context_data = result["result"]
            assert "business_entities" in context_data, "Should have business entities"
            assert "confidence_score" in context_data, "Should have confidence score"
            
            self.test_results.append({
                "test": "Knowledge Graph Integration",
                "status": "PASSED",
                "details": f"Context extracted with {context_data.get('business_entities', 0)} business entities"
            })
            logger.info("✅ Knowledge Graph Integration: PASSED")
            
        except Exception as e:
            self.test_results.append({
                "test": "Knowledge Graph Integration",
                "status": "FAILED",
                "error": str(e)
            })
            logger.error(f"❌ Knowledge Graph Integration: FAILED - {e}")
    
    async def test_multilingual_support(self):
        """Test multilingual decision support capabilities."""
        logger.info("🌍 Test 3: Multilingual Support")
        
        try:
            # Test English decision context
            en_result = await call_unified_mcp_tool(
                "query_decision_context",
                {
                    "content": "We need to improve our market position and increase customer satisfaction.",
                    "language": "en"
                }
            )
            
            # Test Chinese decision context
            zh_result = await call_unified_mcp_tool(
                "query_decision_context",
                {
                    "content": "我们需要改善市场地位并提高客户满意度。",
                    "language": "zh"
                }
            )
            
            # Test Japanese decision context
            ja_result = await call_unified_mcp_tool(
                "query_decision_context",
                {
                    "content": "市場地位を改善し、顧客満足度を向上させる必要があります。",
                    "language": "ja"
                }
            )
            
            assert en_result["success"], "English context extraction failed"
            assert zh_result["success"], "Chinese context extraction failed"
            assert ja_result["success"], "Japanese context extraction failed"
            
            self.test_results.append({
                "test": "Multilingual Support",
                "status": "PASSED",
                "details": "Successfully tested English, Chinese, and Japanese decision contexts"
            })
            logger.info("✅ Multilingual Support: PASSED")
            
        except Exception as e:
            self.test_results.append({
                "test": "Multilingual Support",
                "status": "FAILED",
                "error": str(e)
            })
            logger.error(f"❌ Multilingual Support: FAILED - {e}")
    
    async def test_decision_context_extraction(self):
        """Test decision context extraction from various content types."""
        logger.info("🧠 Test 4: Decision Context Extraction")
        
        try:
            # Test business strategy content
            strategy_content = """
            Our company is facing increased competition in the technology sector.
            We need to develop a new product strategy to maintain market leadership.
            Key challenges include budget constraints and timeline pressure.
            Opportunities exist in emerging markets and new technologies.
            """
            
            result = await call_unified_mcp_tool(
                "query_decision_context",
                {
                    "content": strategy_content,
                    "language": "en",
                    "context_type": "comprehensive"
                }
            )
            
            assert result["success"], "Decision context extraction failed"
            
            context_data = result["result"]
            assert context_data["confidence_score"] > 0, "Should have positive confidence score"
            
            self.test_results.append({
                "test": "Decision Context Extraction",
                "status": "PASSED",
                "details": f"Context extracted with confidence score: {context_data['confidence_score']:.2f}"
            })
            logger.info("✅ Decision Context Extraction: PASSED")
            
        except Exception as e:
            self.test_results.append({
                "test": "Decision Context Extraction",
                "status": "FAILED",
                "error": str(e)
            })
            logger.error(f"❌ Decision Context Extraction: FAILED - {e}")
    
    async def test_entity_extraction_for_decisions(self):
        """Test entity extraction specifically for decision support."""
        logger.info("🏷️ Test 5: Entity Extraction for Decisions")
        
        try:
            content = """
            Microsoft Corporation is expanding into cloud computing services.
            They face competition from Amazon Web Services and Google Cloud.
            The market opportunity is estimated at $500 billion by 2025.
            Key risks include regulatory challenges and technology disruption.
            """
            
            result = await call_unified_mcp_tool(
                "extract_entities_for_decisions",
                {
                    "content": content,
                    "language": "en",
                    "entity_types": ["ORGANIZATION", "MARKET", "RISK", "OPPORTUNITY"]
                }
            )
            
            assert result["success"], "Entity extraction failed"
            
            entities_data = result["result"]
            assert entities_data["count"] > 0, "Should extract at least one entity"
            
            self.test_results.append({
                "test": "Entity Extraction for Decisions",
                "status": "PASSED",
                "details": f"Extracted {entities_data['count']} entities for decision support"
            })
            logger.info("✅ Entity Extraction for Decisions: PASSED")
            
        except Exception as e:
            self.test_results.append({
                "test": "Entity Extraction for Decisions",
                "status": "FAILED",
                "error": str(e)
            })
            logger.error(f"❌ Entity Extraction for Decisions: FAILED - {e}")
    
    async def test_decision_pattern_analysis(self):
        """Test decision pattern analysis capabilities."""
        logger.info("📊 Test 6: Decision Pattern Analysis")
        
        try:
            result = await call_unified_mcp_tool(
                "analyze_decision_patterns",
                {
                    "entity_name": "Microsoft",
                    "pattern_type": "business_patterns",
                    "language": "en",
                    "time_window": "1_year"
                }
            )
            
            assert result["success"], "Pattern analysis failed"
            
            pattern_data = result["result"]
            assert "patterns" in pattern_data, "Should have patterns in result"
            
            self.test_results.append({
                "test": "Decision Pattern Analysis",
                "status": "PASSED",
                "details": f"Analyzed {pattern_data['count']} patterns for Microsoft"
            })
            logger.info("✅ Decision Pattern Analysis: PASSED")
            
        except Exception as e:
            self.test_results.append({
                "test": "Decision Pattern Analysis",
                "status": "FAILED",
                "error": str(e)
            })
            logger.error(f"❌ Decision Pattern Analysis: FAILED - {e}")
    
    async def test_recommendation_generation(self):
        """Test AI-powered recommendation generation."""
        logger.info("💡 Test 7: Recommendation Generation")
        
        try:
            business_context = """
            Our company is a mid-sized technology firm with 500 employees.
            We're experiencing declining market share and increasing operational costs.
            Our current technology stack is becoming outdated.
            We have a budget of $2 million for improvements.
            """
            
            result = await call_unified_mcp_tool(
                "generate_recommendations",
                {
                    "business_context": business_context,
                    "current_performance": {"efficiency": 0.6, "market_share": 0.4},
                    "market_conditions": {"competition": 0.8, "growth_rate": 0.05},
                    "resource_constraints": {"budget": 2000000, "team_size": 50},
                    "language": "en"
                }
            )
            
            assert result["success"], "Recommendation generation failed"
            
            self.test_results.append({
                "test": "Recommendation Generation",
                "status": "PASSED",
                "details": "Successfully generated AI-powered recommendations"
            })
            logger.info("✅ Recommendation Generation: PASSED")
            
        except Exception as e:
            self.test_results.append({
                "test": "Recommendation Generation",
                "status": "FAILED",
                "error": str(e)
            })
            logger.error(f"❌ Recommendation Generation: FAILED - {e}")
    
    async def test_action_prioritization(self):
        """Test action prioritization capabilities."""
        logger.info("⚖️ Test 8: Action Prioritization")
        
        try:
            recommendations = [
                "Implement cloud migration strategy",
                "Upgrade technology infrastructure",
                "Improve customer service processes",
                "Develop new product features",
                "Optimize operational efficiency"
            ]
            
            result = await call_unified_mcp_tool(
                "prioritize_actions",
                {
                    "recommendations": recommendations,
                    "available_resources": {"budget": 2000000, "team_capacity": 50},
                    "time_constraints": {"deadline": "12 months"},
                    "stakeholder_preferences": {"efficiency": 0.8, "innovation": 0.7}
                }
            )
            
            assert result["success"], "Action prioritization failed"
            
            self.test_results.append({
                "test": "Action Prioritization",
                "status": "PASSED",
                "details": "Successfully prioritized actions based on multiple factors"
            })
            logger.info("✅ Action Prioritization: PASSED")
            
        except Exception as e:
            self.test_results.append({
                "test": "Action Prioritization",
                "status": "FAILED",
                "error": str(e)
            })
            logger.error(f"❌ Action Prioritization: FAILED - {e}")
    
    async def test_implementation_planning(self):
        """Test implementation planning capabilities."""
        logger.info("📋 Test 9: Implementation Planning")
        
        try:
            recommendation = "Implement cloud migration strategy to improve scalability and reduce costs"
            
            result = await call_unified_mcp_tool(
                "create_implementation_plans",
                {
                    "recommendation": recommendation,
                    "available_resources": {"budget": 1000000, "team_size": 20},
                    "budget_constraints": 1000000,
                    "timeline_constraints": 180
                }
            )
            
            assert result["success"], "Implementation planning failed"
            
            self.test_results.append({
                "test": "Implementation Planning",
                "status": "PASSED",
                "details": "Successfully created detailed implementation plan"
            })
            logger.info("✅ Implementation Planning: PASSED")
            
        except Exception as e:
            self.test_results.append({
                "test": "Implementation Planning",
                "status": "FAILED",
                "error": str(e)
            })
            logger.error(f"❌ Implementation Planning: FAILED - {e}")
    
    async def test_success_prediction(self):
        """Test success prediction capabilities."""
        logger.info("🔮 Test 10: Success Prediction")
        
        try:
            recommendation = "Implement cloud migration strategy to improve scalability and reduce costs"
            
            result = await call_unified_mcp_tool(
                "predict_success",
                {
                    "recommendation": recommendation,
                    "historical_data": {"cloud_migration_success_rate": 0.75},
                    "organizational_capabilities": {"technical_expertise": 0.8, "change_management": 0.6},
                    "market_conditions": {"cloud_adoption_rate": 0.85, "competition": 0.7}
                }
            )
            
            assert result["success"], "Success prediction failed"
            
            self.test_results.append({
                "test": "Success Prediction",
                "status": "PASSED",
                "details": "Successfully predicted success likelihood for recommendation"
            })
            logger.info("✅ Success Prediction: PASSED")
            
        except Exception as e:
            self.test_results.append({
                "test": "Success Prediction",
                "status": "FAILED",
                "error": str(e)
            })
            logger.error(f"❌ Success Prediction: FAILED - {e}")
    
    async def generate_test_report(self):
        """Generate comprehensive test report."""
        logger.info("📊 Generating Test Report")
        logger.info("=" * 60)
        
        # Calculate statistics
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["status"] == "PASSED"])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Test duration
        duration = datetime.now() - self.start_time
        
        # Create report
        report = {
            "test_suite": "Enhanced Decision Support System",
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration.total_seconds(),
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate
            },
            "test_results": self.test_results
        }
        
        # Save report
        report_file = f"Results/enhanced_decision_support_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        logger.info(f"📈 Test Summary:")
        logger.info(f"   Total Tests: {total_tests}")
        logger.info(f"   Passed: {passed_tests}")
        logger.info(f"   Failed: {failed_tests}")
        logger.info(f"   Success Rate: {success_rate:.1f}%")
        logger.info(f"   Duration: {duration.total_seconds():.1f} seconds")
        logger.info(f"   Report saved to: {report_file}")
        
        # Print failed tests
        if failed_tests > 0:
            logger.warning("❌ Failed Tests:")
            for result in self.test_results:
                if result["status"] == "FAILED":
                    logger.warning(f"   - {result['test']}: {result.get('error', 'Unknown error')}")
        
        logger.info("=" * 60)
        logger.info("🎉 Enhanced Decision Support System Testing Complete!")


async def main():
    """Main test execution function."""
    # Wait for server to be ready
    logger.info("⏳ Waiting for MCP server to be ready...")
    await asyncio.sleep(60)  # Wait 60 seconds for server to fully load
    
    # Run tests
    tester = EnhancedDecisionSupportTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
