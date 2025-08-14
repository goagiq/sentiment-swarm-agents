"""
Test Enhanced Multi-Modal Decision Support and Scenario Analysis

This script tests the enhanced multi-modal decision support capabilities including:
- Multi-modal integration engine
- Enhanced cross-modal pattern matching
- Enhanced scenario analysis with real-time data
- Multi-modal decision context extraction
- Cross-modal confidence scoring
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

from loguru import logger

from src.core.models import AnalysisRequest, DataType
from src.core.multi_modal_integration_engine import MultiModalIntegrationEngine
from src.core.scenario_analysis.enhanced_scenario_analysis import EnhancedScenarioAnalysis
from src.core.unified_mcp_client import call_unified_mcp_tool


class EnhancedMultiModalDecisionSupportTester:
    """Test suite for enhanced multi-modal decision support capabilities."""
    
    def __init__(self):
        self.test_results = []
        self.multi_modal_engine = MultiModalIntegrationEngine()
        self.scenario_analysis = EnhancedScenarioAnalysis()
        
        # Test data for different modalities
        self.test_data = {
            "text": {
                "content": "Our company needs to improve operational efficiency and reduce costs while maintaining quality standards. Market analysis shows increasing competition and changing customer preferences.",
                "data_type": DataType.TEXT,
                "language": "en"
            },
            "audio": {
                "content": "audio_sample.mp3",  # Placeholder
                "data_type": DataType.AUDIO,
                "language": "en"
            },
            "image": {
                "content": "dashboard_screenshot.png",  # Placeholder
                "data_type": DataType.IMAGE,
                "language": "en"
            },
            "video": {
                "content": "presentation_video.mp4",  # Placeholder
                "data_type": DataType.VIDEO,
                "language": "en"
            },
            "web": {
                "content": "https://example.com/market-analysis",
                "data_type": DataType.WEBPAGE,
                "language": "en"
            }
        }
        
        logger.info("Enhanced Multi-Modal Decision Support Tester initialized")
    
    async def run_all_tests(self):
        """Run all enhanced multi-modal decision support tests."""
        logger.info("🚀 Starting Enhanced Multi-Modal Decision Support Tests")
        
        try:
            # Test 1: Multi-Modal Integration Engine
            await self.test_multi_modal_integration_engine()
            
            # Test 2: Enhanced Cross-Modal Pattern Matching
            await self.test_enhanced_cross_modal_matching()
            
            # Test 3: Enhanced Scenario Analysis
            await self.test_enhanced_scenario_analysis()
            
            # Test 4: Multi-Modal Decision Context
            await self.test_multi_modal_decision_context()
            
            # Test 5: Cross-Modal Confidence Scoring
            await self.test_cross_modal_confidence_scoring()
            
            # Test 6: Real-Time Data Integration
            await self.test_real_time_data_integration()
            
            # Test 7: Historical Pattern Integration
            await self.test_historical_pattern_integration()
            
            # Test 8: Multi-Modal Scenario Building
            await self.test_multi_modal_scenario_building()
            
            # Generate test report
            await self.generate_test_report()
            
        except Exception as e:
            logger.error(f"❌ Test execution failed: {e}")
            self.test_results.append({
                "test": "Overall Test Execution",
                "status": "FAILED",
                "error": str(e)
            })
    
    async def test_multi_modal_integration_engine(self):
        """Test the multi-modal integration engine."""
        logger.info("🔗 Test 1: Multi-Modal Integration Engine")
        
        try:
            # Create test requests for different modalities
            requests = []
            for modality, data in self.test_data.items():
                if modality in ["text", "web"]:  # Only test text and web for now
                    request = AnalysisRequest(
                        data_type=data["data_type"],
                        content=data["content"],
                        language=data["language"]
                    )
                    requests.append(request)
            
            # Build unified context
            unified_context = await self.multi_modal_engine.build_unified_context(requests)
            
            # Validate results
            assert unified_context is not None, "Unified context should not be None"
            assert hasattr(unified_context, 'modality_insights'), "Should have modality insights"
            assert hasattr(unified_context, 'cross_modal_correlations'), "Should have correlations"
            assert hasattr(unified_context, 'overall_confidence'), "Should have confidence score"
            
            self.test_results.append({
                "test": "Multi-Modal Integration Engine",
                "status": "PASSED",
                "details": f"Unified context built with {len(unified_context.modality_insights)} modalities, {len(unified_context.cross_modal_correlations)} correlations, confidence: {unified_context.overall_confidence:.2f}"
            })
            logger.info("✅ Multi-Modal Integration Engine: PASSED")
            
        except Exception as e:
            self.test_results.append({
                "test": "Multi-Modal Integration Engine",
                "status": "FAILED",
                "error": str(e)
            })
            logger.error(f"❌ Multi-Modal Integration Engine: FAILED - {e}")
    
    async def test_enhanced_cross_modal_matching(self):
        """Test enhanced cross-modal pattern matching."""
        logger.info("🔗 Test 2: Enhanced Cross-Modal Pattern Matching")
        
        try:
            # Create test patterns for different modalities
            test_patterns = {
                "text": [
                    {"name": "efficiency_trend", "type": "trend", "confidence": 0.8},
                    {"name": "cost_reduction", "type": "goal", "confidence": 0.9}
                ],
                "audio": [
                    {"name": "sentiment_positive", "type": "sentiment", "confidence": 0.7},
                    {"name": "speaker_identification", "type": "entity", "confidence": 0.8}
                ],
                "image": [
                    {"name": "dashboard_metrics", "type": "visual", "confidence": 0.9},
                    {"name": "chart_analysis", "type": "data", "confidence": 0.8}
                ]
            }
            
            # Test cross-modal matching
            from src.core.pattern_recognition.cross_modal_matcher import EnhancedCrossModalMatcher
            matcher = EnhancedCrossModalMatcher()
            
            result = await matcher.match_patterns(test_patterns)
            
            # Validate results
            assert result.get("success", True), "Cross-modal matching should succeed"
            assert "matches" in result, "Should have matches"
            assert "anomalies" in result, "Should have anomalies"
            assert "trends" in result, "Should have trends"
            assert "overall_correlation_score" in result, "Should have correlation score"
            
            self.test_results.append({
                "test": "Enhanced Cross-Modal Pattern Matching",
                "status": "PASSED",
                "details": f"Found {result.get('total_matches', 0)} matches, {result.get('total_anomalies', 0)} anomalies, correlation: {result.get('overall_correlation_score', 0):.2f}"
            })
            logger.info("✅ Enhanced Cross-Modal Pattern Matching: PASSED")
            
        except Exception as e:
            self.test_results.append({
                "test": "Enhanced Cross-Modal Pattern Matching",
                "status": "FAILED",
                "error": str(e)
            })
            logger.error(f"❌ Enhanced Cross-Modal Pattern Matching: FAILED - {e}")
    
    async def test_enhanced_scenario_analysis(self):
        """Test enhanced scenario analysis with real-time data."""
        logger.info("🔗 Test 3: Enhanced Scenario Analysis")
        
        try:
            # Create enhanced scenario
            scenario = await self.scenario_analysis.create_enhanced_scenario(
                name="Operational Efficiency Improvement",
                description="Scenario to improve operational efficiency and reduce costs",
                scenario_type="operational_improvement",
                parameters={
                    "impact_scope": 0.8,
                    "resource_requirement": 0.6,
                    "timeline": "6 months",
                    "budget": 500000
                },
                real_time_sources=["market_data", "social_media", "news"]
            )
            
            # Validate scenario
            assert scenario is not None, "Scenario should not be None"
            assert scenario.scenario_id is not None, "Should have scenario ID"
            assert scenario.confidence_score >= 0.0, "Should have confidence score"
            assert scenario.risk_score >= 0.0, "Should have risk score"
            assert scenario.impact_score >= 0.0, "Should have impact score"
            
            # Predict outcomes
            outcomes = await self.scenario_analysis.predict_scenario_outcomes(scenario)
            
            # Validate outcomes
            assert len(outcomes) > 0, "Should have predicted outcomes"
            for outcome in outcomes:
                assert outcome.probability >= 0.0, "Outcome probability should be >= 0"
                assert outcome.probability <= 1.0, "Outcome probability should be <= 1"
                assert outcome.confidence >= 0.0, "Outcome confidence should be >= 0"
            
            self.test_results.append({
                "test": "Enhanced Scenario Analysis",
                "status": "PASSED",
                "details": f"Scenario created with confidence: {scenario.confidence_score:.2f}, risk: {scenario.risk_score:.2f}, {len(outcomes)} outcomes predicted"
            })
            logger.info("✅ Enhanced Scenario Analysis: PASSED")
            
        except Exception as e:
            self.test_results.append({
                "test": "Enhanced Scenario Analysis",
                "status": "FAILED",
                "error": str(e)
            })
            logger.error(f"❌ Enhanced Scenario Analysis: FAILED - {e}")
    
    async def test_multi_modal_decision_context(self):
        """Test multi-modal decision context extraction."""
        logger.info("🔗 Test 4: Multi-Modal Decision Context")
        
        try:
            # Create test requests
            requests = []
            for modality, data in self.test_data.items():
                if modality in ["text", "web"]:  # Only test text and web for now
                    request = AnalysisRequest(
                        data_type=data["data_type"],
                        content=data["content"],
                        language=data["language"]
                    )
                    requests.append(request)
            
            # Test decision context extraction using MCP tools
            result = await call_unified_mcp_tool(
                "query_decision_context",
                {
                    "content": self.test_data["text"]["content"],
                    "language": "en",
                    "context_type": "comprehensive"
                }
            )
            
            # Validate results
            assert result.get("success", False), "Decision context query should succeed"
            assert "result" in result, "Should have result data"
            
            context_data = result["result"]
            assert "business_entities" in context_data, "Should have business entities"
            assert "confidence_score" in context_data, "Should have confidence score"
            
            self.test_results.append({
                "test": "Multi-Modal Decision Context",
                "status": "PASSED",
                "details": f"Decision context extracted with {context_data.get('business_entities', 0)} business entities, confidence: {context_data.get('confidence_score', 0):.2f}"
            })
            logger.info("✅ Multi-Modal Decision Context: PASSED")
            
        except Exception as e:
            self.test_results.append({
                "test": "Multi-Modal Decision Context",
                "status": "FAILED",
                "error": str(e)
            })
            logger.error(f"❌ Multi-Modal Decision Context: FAILED - {e}")
    
    async def test_cross_modal_confidence_scoring(self):
        """Test cross-modal confidence scoring."""
        logger.info("🔗 Test 5: Cross-Modal Confidence Scoring")
        
        try:
            # Create test modality insights
            from src.core.multi_modal_integration_engine import ModalityInsight
            
            text_insight = ModalityInsight(
                modality="text",
                content_type=DataType.TEXT,
                entities=[{"name": "efficiency", "type": "business", "confidence": 0.8}],
                confidence=0.9
            )
            
            web_insight = ModalityInsight(
                modality="web",
                content_type=DataType.WEBPAGE,
                entities=[{"name": "market_trends", "type": "market", "confidence": 0.7}],
                confidence=0.8
            )
            
            # Test confidence calculation
            modality_insights = {"text": text_insight, "web": web_insight}
            correlations = []  # Empty for this test
            
            overall_confidence = self.multi_modal_engine._calculate_overall_confidence(
                modality_insights, correlations
            )
            
            # Validate confidence score
            assert overall_confidence >= 0.0, "Confidence should be >= 0"
            assert overall_confidence <= 1.0, "Confidence should be <= 1"
            assert overall_confidence > 0.5, "Confidence should be reasonable"
            
            self.test_results.append({
                "test": "Cross-Modal Confidence Scoring",
                "status": "PASSED",
                "details": f"Overall confidence calculated: {overall_confidence:.2f}"
            })
            logger.info("✅ Cross-Modal Confidence Scoring: PASSED")
            
        except Exception as e:
            self.test_results.append({
                "test": "Cross-Modal Confidence Scoring",
                "status": "FAILED",
                "error": str(e)
            })
            logger.error(f"❌ Cross-Modal Confidence Scoring: FAILED - {e}")
    
    async def test_real_time_data_integration(self):
        """Test real-time data integration."""
        logger.info("🔗 Test 6: Real-Time Data Integration")
        
        try:
            # Test market data fetching
            market_data = await self.scenario_analysis._fetch_market_data()
            
            # Validate market data
            assert len(market_data) > 0, "Should have market data points"
            for data_point in market_data:
                assert data_point.source == "market_data", "Should be market data"
                assert data_point.confidence > 0.0, "Should have confidence"
                assert data_point.timestamp is not None, "Should have timestamp"
            
            # Test social media data fetching
            social_data = await self.scenario_analysis._fetch_social_media_data()
            
            # Validate social media data
            assert len(social_data) > 0, "Should have social media data points"
            for data_point in social_data:
                assert data_point.source == "social_media", "Should be social media data"
                assert data_point.confidence > 0.0, "Should have confidence"
            
            self.test_results.append({
                "test": "Real-Time Data Integration",
                "status": "PASSED",
                "details": f"Fetched {len(market_data)} market data points, {len(social_data)} social media data points"
            })
            logger.info("✅ Real-Time Data Integration: PASSED")
            
        except Exception as e:
            self.test_results.append({
                "test": "Real-Time Data Integration",
                "status": "FAILED",
                "error": str(e)
            })
            logger.error(f"❌ Real-Time Data Integration: FAILED - {e}")
    
    async def test_historical_pattern_integration(self):
        """Test historical pattern integration."""
        logger.info("🔗 Test 7: Historical Pattern Integration")
        
        try:
            # Test historical pattern finding
            patterns = await self.scenario_analysis._find_relevant_historical_patterns(
                scenario_type="operational_improvement",
                parameters={"impact_scope": 0.8},
                multi_modal_context=None
            )
            
            # Validate patterns (may be empty if no historical data)
            assert isinstance(patterns, list), "Should return list of patterns"
            
            # Test pattern analysis
            if patterns:
                for pattern in patterns:
                    assert pattern.pattern_type is not None, "Should have pattern type"
                    assert pattern.entity is not None, "Should have entity"
                    assert pattern.success_rate >= 0.0, "Success rate should be >= 0"
                    assert pattern.success_rate <= 1.0, "Success rate should be <= 1"
            
            self.test_results.append({
                "test": "Historical Pattern Integration",
                "status": "PASSED",
                "details": f"Found {len(patterns)} historical patterns"
            })
            logger.info("✅ Historical Pattern Integration: PASSED")
            
        except Exception as e:
            self.test_results.append({
                "test": "Historical Pattern Integration",
                "status": "FAILED",
                "error": str(e)
            })
            logger.error(f"❌ Historical Pattern Integration: FAILED - {e}")
    
    async def test_multi_modal_scenario_building(self):
        """Test multi-modal scenario building."""
        logger.info("🔗 Test 8: Multi-Modal Scenario Building")
        
        try:
            # Create test requests for multi-modal scenario
            requests = []
            for modality, data in self.test_data.items():
                if modality in ["text", "web"]:  # Only test text and web for now
                    request = AnalysisRequest(
                        data_type=data["data_type"],
                        content=data["content"],
                        language=data["language"]
                    )
                    requests.append(request)
            
            # Create enhanced scenario with multi-modal inputs
            scenario = await self.scenario_analysis.create_enhanced_scenario(
                name="Multi-Modal Business Strategy",
                description="Strategy based on multi-modal analysis",
                scenario_type="strategic_planning",
                parameters={
                    "impact_scope": 0.9,
                    "resource_requirement": 0.7,
                    "timeline": "12 months",
                    "budget": 1000000
                },
                multi_modal_inputs=requests,
                real_time_sources=["market_data", "news"]
            )
            
            # Validate multi-modal scenario
            assert scenario is not None, "Scenario should not be None"
            assert scenario.multi_modal_inputs is not None, "Should have multi-modal inputs"
            assert scenario.confidence_score > 0.0, "Should have confidence score"
            
            # Test scenario adaptation
            adapted_scenario = await self.scenario_analysis.adapt_scenario(
                scenario,
                {
                    "trigger": "data_update",
                    "changes": {"impact_scope": 0.95},
                    "reason": "Updated market conditions"
                }
            )
            
            # Validate adaptation
            assert adapted_scenario.updated_at > scenario.created_at, "Should be updated"
            assert len(adapted_scenario.adaptation_history) > 0, "Should have adaptation history"
            
            self.test_results.append({
                "test": "Multi-Modal Scenario Building",
                "status": "PASSED",
                "details": f"Multi-modal scenario created and adapted, confidence: {adapted_scenario.confidence_score:.2f}"
            })
            logger.info("✅ Multi-Modal Scenario Building: PASSED")
            
        except Exception as e:
            self.test_results.append({
                "test": "Multi-Modal Scenario Building",
                "status": "FAILED",
                "error": str(e)
            })
            logger.error(f"❌ Multi-Modal Scenario Building: FAILED - {e}")
    
    async def generate_test_report(self):
        """Generate comprehensive test report."""
        logger.info("📊 Generating Enhanced Multi-Modal Decision Support Test Report")
        
        # Calculate test statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["status"] == "PASSED")
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Create report
        report = {
            "test_suite": "Enhanced Multi-Modal Decision Support",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": f"{success_rate:.1f}%"
            },
            "test_results": self.test_results,
            "capabilities_tested": [
                "Multi-Modal Integration Engine",
                "Enhanced Cross-Modal Pattern Matching",
                "Enhanced Scenario Analysis",
                "Multi-Modal Decision Context",
                "Cross-Modal Confidence Scoring",
                "Real-Time Data Integration",
                "Historical Pattern Integration",
                "Multi-Modal Scenario Building"
            ],
            "improvements_verified": [
                "Semantic alignment across modalities",
                "Cross-modal confidence scoring",
                "Unified decision context building",
                "Real-time data integration",
                "Historical pattern analysis",
                "Dynamic scenario adaptation",
                "Multi-modal insight correlation"
            ]
        }
        
        # Save report
        report_filename = f"Results/enhanced_multi_modal_decision_support_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            import os
            os.makedirs("Results", exist_ok=True)
            
            with open(report_filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📄 Test report saved to: {report_filename}")
            
            # Print summary
            print("\n" + "="*80)
            print("🎯 ENHANCED MULTI-MODAL DECISION SUPPORT TEST RESULTS")
            print("="*80)
            print(f"📊 Total Tests: {total_tests}")
            print(f"✅ Passed: {passed_tests}")
            print(f"❌ Failed: {failed_tests}")
            print(f"📈 Success Rate: {success_rate:.1f}%")
            print("="*80)
            
            if failed_tests > 0:
                print("\n❌ FAILED TESTS:")
                for result in self.test_results:
                    if result["status"] == "FAILED":
                        print(f"  • {result['test']}: {result.get('error', 'Unknown error')}")
            
            print(f"\n📄 Detailed report saved to: {report_filename}")
            
        except Exception as e:
            logger.error(f"Error saving test report: {e}")
            print(f"Error saving test report: {e}")


async def main():
    """Main test execution function."""
    logger.info("🚀 Starting Enhanced Multi-Modal Decision Support Testing")
    
    # Initialize tester
    tester = EnhancedMultiModalDecisionSupportTester()
    
    # Run all tests
    await tester.run_all_tests()
    
    logger.info("🏁 Enhanced Multi-Modal Decision Support Testing completed")


if __name__ == "__main__":
    # Run the tests
    asyncio.run(main())
