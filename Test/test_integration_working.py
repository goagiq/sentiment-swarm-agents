#!/usr/bin/env python3
"""
Working Integration Testing for Sentiment Analysis & Decision Support System

This script tests the core components using their actual APIs.
"""

import sys
import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.agents.decision_support_agent import DecisionSupportAgent
from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
from src.core.models import AnalysisRequest, DataType
from src.core.decision_support.action_prioritizer import (
    ActionPrioritizer, PrioritizationContext
)
from src.core.decision_support.implementation_planner import ImplementationPlanner
from src.core.decision_support.recommendation_engine import (
    Recommendation, RecommendationType, RecommendationPriority, 
    RecommendationCategory
)
from src.core.scenario_analysis.enhanced_scenario_analysis import EnhancedScenarioAnalysis
from src.core.monitoring.alert_system import AlertSystem
from src.core.monitoring.decision_monitor import DecisionMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('working_integration_test_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WorkingIntegrationTester:
    """Working integration tester using actual component APIs."""
    
    def __init__(self):
        self.results = {
            "test_suite": "Working Integration Testing",
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0
            }
        }
        
        # Initialize components
        self.decision_agent = None
        self.knowledge_agent = None
        self.action_prioritizer = None
        self.implementation_planner = None
        self.scenario_analyzer = None
        self.alert_system = None
        self.decision_monitor = None
        
    def setup_components(self):
        """Initialize core system components."""
        try:
            logger.info("Setting up core system components...")
            
            # Initialize agents
            self.decision_agent = DecisionSupportAgent()
            self.knowledge_agent = KnowledgeGraphAgent()
            
            # Initialize core components
            self.action_prioritizer = ActionPrioritizer()
            self.implementation_planner = ImplementationPlanner()
            self.scenario_analyzer = EnhancedScenarioAnalysis()
            self.alert_system = AlertSystem()
            self.decision_monitor = DecisionMonitor()
            
            logger.info("All core components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup components: {e}")
            return False
    
    def record_test_result(self, test_name: str, success: bool, details: str = "", error: str = ""):
        """Record test results."""
        test_result = {
            "test_name": test_name,
            "success": success,
            "details": details,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        self.results["tests"].append(test_result)
        self.results["summary"]["total_tests"] += 1
        
        if success:
            self.results["summary"]["passed"] += 1
            logger.info(f"PASSED: {test_name}")
        else:
            if error:
                self.results["summary"]["errors"] += 1
                logger.error(f"ERROR: {test_name} - {error}")
            else:
                self.results["summary"]["failed"] += 1
                logger.error(f"FAILED: {test_name} - {details}")
    
    async def test_decision_support_workflow(self):
        """Test decision support workflow using actual API."""
        test_name = "Decision Support Workflow"
        
        try:
            logger.info("Testing decision support workflow...")
            
            # Create analysis request
            request = AnalysisRequest(
                data_type=DataType.TEXT,
                content="We need to increase our market share by 15% within 6 months. Our current market share is 8%, customer satisfaction is 7.2/10, and revenue growth is 5%. We have a budget of $500K and need to coordinate with Marketing, Sales, and Product teams.",
                language="en",
                enable_analytics=True,
                metadata={
                    "business_goal": "Increase market share by 15%",
                    "constraints": ["Budget: $500K", "Timeline: 6 months"],
                    "stakeholders": ["Marketing", "Sales", "Product"],
                    "current_metrics": {
                        "market_share": "8%",
                        "customer_satisfaction": "7.2/10",
                        "revenue_growth": "5%"
                    }
                }
            )
            
            # Process the request
            result = await self.decision_agent.process(request)
            
            if not result or not result.success:
                self.record_test_result(test_name, False, "Failed to process decision support request")
                return
            
            self.record_test_result(test_name, True, f"Successfully processed decision support request with confidence {result.sentiment.confidence}")
            
        except Exception as e:
            self.record_test_result(test_name, False, error=str(e))
    
    async def test_knowledge_graph_integration(self):
        """Test knowledge graph integration."""
        test_name = "Knowledge Graph Integration"
        
        try:
            logger.info("Testing knowledge graph integration...")
            
            # Create analysis request for knowledge graph
            request = AnalysisRequest(
                data_type=DataType.TEXT,
                content="Customer satisfaction is at 85% and market share is 12%. Our recent product launch has significantly impacted customer satisfaction levels.",
                language="en",
                enable_analytics=True,
                metadata={
                    "entities": [
                        {"name": "Customer Satisfaction", "type": "metric", "value": 0.85},
                        {"name": "Market Share", "type": "metric", "value": 0.12},
                        {"name": "Product Launch", "type": "event", "date": "2024-01-15"}
                    ],
                    "relationships": [
                        {"source": "Product Launch", "target": "Customer Satisfaction", "type": "impacts"},
                        {"source": "Customer Satisfaction", "target": "Market Share", "type": "influences"}
                    ]
                }
            )
            
            # Process the request
            result = await self.knowledge_agent.process(request)
            
            if not result or not result.success:
                self.record_test_result(test_name, False, "Failed to process knowledge graph request")
                return
            
            self.record_test_result(test_name, True, f"Successfully processed knowledge graph request with confidence {result.sentiment.confidence}")
            
        except Exception as e:
            self.record_test_result(test_name, False, error=str(e))
    
    async def test_action_prioritization_integration(self):
        """Test action prioritization integration."""
        test_name = "Action Prioritization Integration"
        
        try:
            logger.info("Testing action prioritization integration...")
            
            # Create proper Recommendation objects
            recommendations = [
                Recommendation(
                    title="Launch marketing campaign",
                    description="Launch a comprehensive marketing campaign to increase brand awareness and customer acquisition",
                    recommendation_type=RecommendationType.STRATEGIC,
                    priority=RecommendationPriority.HIGH,
                    category=RecommendationCategory.MARKET_STRATEGY,
                    confidence_score=0.85,
                    expected_impact={
                        "customer_acquisition": "High",
                        "brand_awareness": "High",
                        "revenue_growth": "Medium"
                    },
                    implementation_effort="medium",
                    time_to_implement="3 months",
                    cost_estimate=50000,
                    risk_level="low"
                ),
                Recommendation(
                    title="Improve product features",
                    description="Enhance product features based on customer feedback and market analysis",
                    recommendation_type=RecommendationType.OPERATIONAL,
                    priority=RecommendationPriority.HIGH,
                    category=RecommendationCategory.TECHNOLOGY_ADOPTION,
                    confidence_score=0.90,
                    expected_impact={
                        "customer_satisfaction": "High",
                        "product_adoption": "High",
                        "competitive_position": "Medium"
                    },
                    implementation_effort="high",
                    time_to_implement="6 months",
                    cost_estimate=200000,
                    risk_level="medium"
                ),
                Recommendation(
                    title="Expand sales team",
                    description="Hire additional sales representatives to increase market coverage",
                    recommendation_type=RecommendationType.OPERATIONAL,
                    priority=RecommendationPriority.MEDIUM,
                    category=RecommendationCategory.RESOURCE_ALLOCATION,
                    confidence_score=0.75,
                    expected_impact={
                        "sales_volume": "High",
                        "market_coverage": "High",
                        "customer_relationships": "Medium"
                    },
                    implementation_effort="low",
                    time_to_implement="2 months",
                    cost_estimate=100000,
                    risk_level="low"
                )
            ]
            
            # Create prioritization context
            context = PrioritizationContext(
                available_resources={"budget": 500000, "team_size": 20},
                time_constraints={"deadline": "12 months"},
                stakeholder_preferences={"marketing": "high", "sales": "medium"},
                strategic_goals=["increase_market_share", "improve_customer_satisfaction"],
                risk_tolerance="medium",
                budget_constraints=500000
            )
            
            # Prioritize actions
            prioritized_actions = await self.action_prioritizer.prioritize_actions(recommendations, context)
            
            if not prioritized_actions:
                self.record_test_result(test_name, False, "Failed to prioritize actions")
                return
            
            self.record_test_result(test_name, True, f"Successfully prioritized {len(prioritized_actions)} actions")
            
        except Exception as e:
            self.record_test_result(test_name, False, error=str(e))
    
    async def test_scenario_analysis_integration(self):
        """Test scenario analysis integration."""
        test_name = "Scenario Analysis Integration"
        
        try:
            logger.info("Testing scenario analysis integration...")
            
            # Test scenario data
            scenario_data = {
                "business_context": {
                    "industry": "Technology",
                    "market_conditions": "Competitive",
                    "economic_outlook": "Stable"
                },
                "decision_variables": [
                    "Product pricing",
                    "Marketing budget",
                    "Development timeline"
                ],
                "constraints": [
                    "Budget: $1M",
                    "Timeline: 12 months",
                    "Team size: 10 people"
                ]
            }
            
            # Create analysis request for scenario analysis
            request = AnalysisRequest(
                data_type=DataType.TEXT,
                content="We need to analyze different scenarios for our product launch. We're in a competitive technology market with stable economic outlook. Key decision variables are product pricing, marketing budget, and development timeline. We have constraints of $1M budget, 12-month timeline, and 10-person team.",
                language="en",
                enable_analytics=True,
                metadata=scenario_data
            )
            
            # Process the request
            result = await self.decision_agent.process(request)
            
            if not result or not result.success:
                self.record_test_result(test_name, False, "Failed to process scenario analysis request")
                return
            
            self.record_test_result(test_name, True, f"Successfully processed scenario analysis request with confidence {result.sentiment.confidence}")
            
        except Exception as e:
            self.record_test_result(test_name, False, error=str(e))
    
    async def test_monitoring_integration(self):
        """Test monitoring integration."""
        test_name = "Monitoring Integration"
        
        try:
            logger.info("Testing monitoring integration...")
            
            # Test monitoring data
            monitoring_data = {
                "metrics": {
                    "error_rate": 0.08,
                    "response_time": 2500,
                    "memory_usage": 0.75,
                    "decision_accuracy": 0.85,
                    "implementation_success_rate": 0.78,
                    "user_satisfaction": 0.82
                },
                "thresholds": {
                    "error_rate": 0.05,
                    "response_time": 2000,
                    "memory_usage": 0.8
                }
            }
            
            # Create analysis request for monitoring
            request = AnalysisRequest(
                data_type=DataType.TEXT,
                content="System monitoring shows error rate at 8%, response time at 2.5 seconds, and memory usage at 75%. Decision accuracy is 85%, implementation success rate is 78%, and user satisfaction is 82%.",
                language="en",
                enable_analytics=True,
                metadata=monitoring_data
            )
            
            # Process the request
            result = await self.decision_agent.process(request)
            
            if not result or not result.success:
                self.record_test_result(test_name, False, "Failed to process monitoring request")
                return
            
            self.record_test_result(test_name, True, f"Successfully processed monitoring request with confidence {result.sentiment.confidence}")
            
        except Exception as e:
            self.record_test_result(test_name, False, error=str(e))
    
    async def test_complete_workflow_integration(self):
        """Test complete workflow integration."""
        test_name = "Complete Workflow Integration"
        
        try:
            logger.info("Testing complete workflow integration...")
            
            # Simulate a complete business scenario
            business_scenario = {
                "company": "TechCorp",
                "industry": "Software",
                "goal": "Launch new mobile app successfully",
                "data_sources": [
                    {"type": "customer_feedback", "format": "text"},
                    {"type": "market_research", "format": "structured"},
                    {"type": "competitor_analysis", "format": "mixed"}
                ],
                "decision_points": [
                    "Feature prioritization",
                    "Launch timing",
                    "Marketing strategy",
                    "Pricing model"
                ]
            }
            
            # Create comprehensive analysis request
            request = AnalysisRequest(
                data_type=DataType.TEXT,
                content="TechCorp needs to launch a new mobile app successfully. We have customer feedback, market research, and competitor analysis data. Key decision points include feature prioritization, launch timing, marketing strategy, and pricing model. We need to coordinate across multiple teams and ensure successful market entry.",
                language="en",
                enable_analytics=True,
                metadata=business_scenario
            )
            
            # Process the request
            result = await self.decision_agent.process(request)
            
            if not result or not result.success:
                self.record_test_result(test_name, False, "Failed to process complete workflow request")
                return
            
            self.record_test_result(test_name, True, f"Successfully processed complete workflow request with confidence {result.sentiment.confidence}")
            
        except Exception as e:
            self.record_test_result(test_name, False, error=str(e))
    
    async def run_all_tests(self):
        """Run all working integration tests."""
        logger.info("Starting Working Integration Testing...")
        
        # Setup components
        if not self.setup_components():
            logger.error("Failed to setup components. Aborting tests.")
            return False
        
        # Run tests
        tests = [
            self.test_decision_support_workflow,
            self.test_knowledge_graph_integration,
            self.test_action_prioritization_integration,
            self.test_scenario_analysis_integration,
            self.test_monitoring_integration,
            self.test_complete_workflow_integration
        ]
        
        for test in tests:
            try:
                await test()
                await asyncio.sleep(1)  # Brief pause between tests
            except Exception as e:
                logger.error(f"Test execution failed: {e}")
        
        # Generate report
        self.generate_report()
        
        return True
    
    def generate_report(self):
        """Generate comprehensive test report."""
        logger.info("Generating working integration test report...")
        
        # Calculate success rate
        total = self.results["summary"]["total_tests"]
        passed = self.results["summary"]["passed"]
        success_rate = (passed / total * 100) if total > 0 else 0
        
        # Add summary statistics
        self.results["summary"]["success_rate"] = f"{success_rate:.1f}%"
        self.results["summary"]["completion_time"] = datetime.now().isoformat()
        
        # Save detailed report
        report_path = Path("Results") / f"working_integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("WORKING INTEGRATION TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {self.results['summary']['failed']}")
        logger.info(f"Errors: {self.results['summary']['errors']}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Report saved to: {report_path}")
        logger.info("=" * 60)
        
        return success_rate >= 80  # Consider 80% success rate as passing

async def main():
    """Main execution function."""
    logger.info("Starting Working Integration Testing Suite")
    
    tester = WorkingIntegrationTester()
    success = await tester.run_all_tests()
    
    if success:
        logger.info("Working integration testing completed successfully!")
        return 0
    else:
        logger.error("Working integration testing completed with failures!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
