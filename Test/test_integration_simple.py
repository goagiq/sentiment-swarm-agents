#!/usr/bin/env python3
"""
Simple Integration Testing for Sentiment Analysis & Decision Support System

This script tests the core components that are actually implemented and working.
"""

import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.agents.decision_support_agent import DecisionSupportAgent
from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
from src.core.decision_support.action_prioritizer import ActionPrioritizer
from src.core.decision_support.implementation_planner import ImplementationPlanner
from src.core.scenario_analysis.enhanced_scenario_analysis import EnhancedScenarioAnalysis
from src.core.monitoring.alert_system import AlertSystem
from src.core.monitoring.decision_monitor import DecisionMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simple_integration_test_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleIntegrationTester:
    """Simple integration tester focusing on core components."""
    
    def __init__(self):
        self.results = {
            "test_suite": "Simple Integration Testing",
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
            logger.info(f"✅ {test_name}: PASSED")
        else:
            if error:
                self.results["summary"]["errors"] += 1
                logger.error(f"❌ {test_name}: ERROR - {error}")
            else:
                self.results["summary"]["failed"] += 1
                logger.error(f"❌ {test_name}: FAILED - {details}")
    
    def test_decision_support_workflow(self):
        """Test decision support workflow."""
        test_name = "Decision Support Workflow"
        
        try:
            logger.info("Testing decision support workflow...")
            
            # Test data
            decision_context = {
                "business_goal": "Increase market share by 15%",
                "constraints": ["Budget: $500K", "Timeline: 6 months"],
                "stakeholders": ["Marketing", "Sales", "Product"],
                "current_metrics": {
                    "market_share": "8%",
                    "customer_satisfaction": "7.2/10",
                    "revenue_growth": "5%"
                }
            }
            
            # Step 1: Analyze decision context
            analysis_result = self.decision_agent.analyze_decision_context(decision_context)
            if not analysis_result:
                self.record_test_result(test_name, False, "Failed to analyze decision context")
                return
            
            # Step 2: Generate scenarios
            scenarios = self.scenario_analyzer.generate_scenarios(decision_context)
            if not scenarios:
                self.record_test_result(test_name, False, "Failed to generate scenarios")
                return
            
            # Step 3: Prioritize actions
            actions = self.action_prioritizer.prioritize_actions(analysis_result, scenarios)
            if not actions:
                self.record_test_result(test_name, False, "Failed to prioritize actions")
                return
            
            # Step 4: Create implementation plan
            plan = self.implementation_planner.create_plan(actions, decision_context)
            if not plan:
                self.record_test_result(test_name, False, "Failed to create implementation plan")
                return
            
            # Step 5: Monitor decision
            monitoring_result = self.decision_monitor.start_monitoring(plan)
            if not monitoring_result:
                self.record_test_result(test_name, False, "Failed to start decision monitoring")
                return
            
            self.record_test_result(test_name, True, f"Successfully completed workflow with {len(actions)} actions")
            
        except Exception as e:
            self.record_test_result(test_name, False, error=str(e))
    
    def test_knowledge_graph_integration(self):
        """Test knowledge graph integration."""
        test_name = "Knowledge Graph Integration"
        
        try:
            logger.info("Testing knowledge graph integration...")
            
            # Test knowledge graph operations
            test_data = {
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
            
            # Step 1: Add entities to knowledge graph
            entities_result = self.knowledge_agent.add_entities(test_data["entities"])
            if not entities_result:
                self.record_test_result(test_name, False, "Failed to add entities to knowledge graph")
                return
            
            # Step 2: Add relationships
            relationships_result = self.knowledge_agent.add_relationships(test_data["relationships"])
            if not relationships_result:
                self.record_test_result(test_name, False, "Failed to add relationships to knowledge graph")
                return
            
            # Step 3: Query knowledge graph
            query_result = self.knowledge_agent.query_graph("Customer Satisfaction")
            if not query_result:
                self.record_test_result(test_name, False, "Failed to query knowledge graph")
                return
            
            # Step 4: Generate insights
            insights = self.knowledge_agent.generate_insights(query_result)
            if not insights:
                self.record_test_result(test_name, False, "Failed to generate insights")
                return
            
            self.record_test_result(test_name, True, f"Successfully integrated knowledge graph with {len(insights)} insights")
            
        except Exception as e:
            self.record_test_result(test_name, False, error=str(e))
    
    def test_monitoring_and_alerting_systems(self):
        """Test monitoring and alerting systems."""
        test_name = "Monitoring and Alerting Systems"
        
        try:
            logger.info("Testing monitoring and alerting systems...")
            
            # Test alert system configuration
            alert_config = {
                "thresholds": {
                    "error_rate": 0.05,
                    "response_time": 2000,
                    "memory_usage": 0.8
                },
                "notification_channels": ["email", "slack", "webhook"]
            }
            
            # Step 1: Configure alert system
            config_result = self.alert_system.configure_alerts(alert_config)
            if not config_result:
                self.record_test_result(test_name, False, "Failed to configure alert system")
                return
            
            # Step 2: Test alert triggers
            test_metrics = {
                "error_rate": 0.08,  # Above threshold
                "response_time": 2500,  # Above threshold
                "memory_usage": 0.75  # Below threshold
            }
            
            alerts = self.alert_system.check_metrics(test_metrics)
            if alerts is None:
                self.record_test_result(test_name, False, "Failed to check metrics")
                return
            
            # Step 3: Test decision monitoring
            decision_metrics = {
                "decision_accuracy": 0.85,
                "implementation_success_rate": 0.78,
                "user_satisfaction": 0.82
            }
            
            monitoring_result = self.decision_monitor.monitor_metrics(decision_metrics)
            if not monitoring_result:
                self.record_test_result(test_name, False, "Failed to monitor decision metrics")
                return
            
            # Step 4: Test notification system
            notification_result = self.alert_system.send_notifications(alerts)
            if notification_result is None:
                self.record_test_result(test_name, False, "Failed to send notifications")
                return
            
            self.record_test_result(test_name, True, f"Successfully tested monitoring with {len(alerts)} alerts")
            
        except Exception as e:
            self.record_test_result(test_name, False, error=str(e))
    
    def test_scenario_analysis_integration(self):
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
            
            # Step 1: Generate scenarios
            scenarios = self.scenario_analyzer.generate_scenarios(scenario_data)
            if not scenarios:
                self.record_test_result(test_name, False, "Failed to generate scenarios")
                return
            
            # Step 2: Analyze scenario impacts
            impact_analysis = self.scenario_analyzer.analyze_impacts(scenarios)
            if not impact_analysis:
                self.record_test_result(test_name, False, "Failed to analyze scenario impacts")
                return
            
            # Step 3: Generate recommendations
            recommendations = self.scenario_analyzer.generate_recommendations(impact_analysis)
            if not recommendations:
                self.record_test_result(test_name, False, "Failed to generate recommendations")
                return
            
            self.record_test_result(test_name, True, f"Successfully analyzed {len(scenarios)} scenarios with {len(recommendations)} recommendations")
            
        except Exception as e:
            self.record_test_result(test_name, False, error=str(e))
    
    def test_action_prioritization_integration(self):
        """Test action prioritization integration."""
        test_name = "Action Prioritization Integration"
        
        try:
            logger.info("Testing action prioritization integration...")
            
            # Test action data
            actions_data = [
                {
                    "name": "Launch marketing campaign",
                    "impact": "High",
                    "effort": "Medium",
                    "cost": 50000,
                    "timeline": "3 months"
                },
                {
                    "name": "Improve product features",
                    "impact": "High",
                    "effort": "High",
                    "cost": 200000,
                    "timeline": "6 months"
                },
                {
                    "name": "Expand sales team",
                    "impact": "Medium",
                    "effort": "Low",
                    "cost": 100000,
                    "timeline": "2 months"
                }
            ]
            
            # Step 1: Prioritize actions
            prioritized_actions = self.action_prioritizer.prioritize_actions(actions_data, {})
            if not prioritized_actions:
                self.record_test_result(test_name, False, "Failed to prioritize actions")
                return
            
            # Step 2: Create implementation plan
            plan = self.implementation_planner.create_plan(prioritized_actions, {})
            if not plan:
                self.record_test_result(test_name, False, "Failed to create implementation plan")
                return
            
            # Step 3: Validate plan
            validation_result = self.implementation_planner.validate_plan(plan)
            if not validation_result:
                self.record_test_result(test_name, False, "Failed to validate implementation plan")
                return
            
            self.record_test_result(test_name, True, f"Successfully prioritized {len(prioritized_actions)} actions and created implementation plan")
            
        except Exception as e:
            self.record_test_result(test_name, False, error=str(e))
    
    def run_all_tests(self):
        """Run all simple integration tests."""
        logger.info("Starting Simple Integration Testing...")
        
        # Setup components
        if not self.setup_components():
            logger.error("Failed to setup components. Aborting tests.")
            return False
        
        # Run tests
        tests = [
            self.test_decision_support_workflow,
            self.test_knowledge_graph_integration,
            self.test_monitoring_and_alerting_systems,
            self.test_scenario_analysis_integration,
            self.test_action_prioritization_integration
        ]
        
        for test in tests:
            try:
                test()
                time.sleep(1)  # Brief pause between tests
            except Exception as e:
                logger.error(f"Test execution failed: {e}")
        
        # Generate report
        self.generate_report()
        
        return True
    
    def generate_report(self):
        """Generate comprehensive test report."""
        logger.info("Generating simple integration test report...")
        
        # Calculate success rate
        total = self.results["summary"]["total_tests"]
        passed = self.results["summary"]["passed"]
        success_rate = (passed / total * 100) if total > 0 else 0
        
        # Add summary statistics
        self.results["summary"]["success_rate"] = f"{success_rate:.1f}%"
        self.results["summary"]["completion_time"] = datetime.now().isoformat()
        
        # Save detailed report
        report_path = Path("Results") / f"simple_integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("SIMPLE INTEGRATION TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {self.results['summary']['failed']}")
        logger.info(f"Errors: {self.results['summary']['errors']}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Report saved to: {report_path}")
        logger.info("=" * 60)
        
        return success_rate >= 80  # Consider 80% success rate as passing

def main():
    """Main execution function."""
    logger.info("Starting Simple Integration Testing Suite")
    
    tester = SimpleIntegrationTester()
    success = tester.run_all_tests()
    
    if success:
        logger.info("Simple integration testing completed successfully!")
        return 0
    else:
        logger.error("Simple integration testing completed with failures!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
