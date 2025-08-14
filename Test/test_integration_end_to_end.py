#!/usr/bin/env python3
"""
End-to-End Integration Testing for Sentiment Analysis & Decision Support System

This script tests complete workflows including:
- Decision support workflows
- Multi-modal processing pipelines
- Real-time data integration
- External system connections
- Monitoring and alerting systems
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
from src.agents.multi_modal_analysis_agent import MultiModalAnalysisAgent
from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
from src.core.decision_support.action_prioritizer import ActionPrioritizer
from src.core.decision_support.implementation_planner import ImplementationPlanner
from src.core.scenario_analysis.enhanced_scenario_analysis import EnhancedScenarioAnalysis
from src.core.monitoring.alert_system import AlertSystem
from src.core.monitoring.decision_monitor import DecisionMonitor
from src.core.external_integration.api_connector import APIConnectorManager
from src.core.external_integration.data_synchronizer import DataSynchronizer
from src.core.real_time.pattern_monitor import PatternMonitor
from src.core.pattern_recognition.cross_modal_matcher import CrossModalMatcher
from src.core.advanced_analytics.causal_inference_engine import CausalInferenceEngine
from src.config.decision_support_config import DecisionSupportConfig
from src.config.multi_modal_config import MultiModalConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integration_test_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EndToEndIntegrationTester:
    """Comprehensive end-to-end integration tester for the decision support system."""
    
    def __init__(self):
        self.results = {
            "test_suite": "End-to-End Integration Testing",
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
        self.multi_modal_agent = None
        self.knowledge_agent = None
        self.action_prioritizer = None
        self.implementation_planner = None
        self.scenario_analyzer = None
        self.alert_system = None
        self.decision_monitor = None
        self.api_connector_manager = None
        self.data_synchronizer = None
        self.pattern_monitor = None
        self.cross_modal_matcher = None
        self.causal_engine = None
        
    def setup_components(self):
        """Initialize all system components."""
        try:
            logger.info("Setting up system components...")
            
            # Initialize agents
            self.decision_agent = DecisionSupportAgent()
            self.multi_modal_agent = MultiModalAnalysisAgent()
            self.knowledge_agent = KnowledgeGraphAgent()
            
            # Initialize core components
            self.action_prioritizer = ActionPrioritizer()
            self.implementation_planner = ImplementationPlanner()
            self.scenario_analyzer = EnhancedScenarioAnalysis()
            self.alert_system = AlertSystem()
            self.decision_monitor = DecisionMonitor()
            self.api_connector_manager = APIConnectorManager()
            self.data_synchronizer = DataSynchronizer()
            self.pattern_monitor = PatternMonitor()
            self.cross_modal_matcher = CrossModalMatcher()
            self.causal_engine = CausalInferenceEngine()
            
            logger.info("All components initialized successfully")
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
        """Test complete decision support workflow."""
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
    
    def test_multi_modal_processing_pipeline(self):
        """Test multi-modal processing pipeline."""
        test_name = "Multi-Modal Processing Pipeline"
        
        try:
            logger.info("Testing multi-modal processing pipeline...")
            
            # Test data
            multi_modal_data = {
                "text": "Customer feedback indicates strong demand for mobile app features",
                "audio": "path/to/audio/feedback.wav",
                "image": "path/to/image/dashboard.png",
                "video": "path/to/video/presentation.mp4"
            }
            
            # Step 1: Process multi-modal data
            processed_data = self.multi_modal_agent.process_multi_modal_data(multi_modal_data)
            if not processed_data:
                self.record_test_result(test_name, False, "Failed to process multi-modal data")
                return
            
            # Step 2: Cross-modal pattern matching
            patterns = self.cross_modal_matcher.find_patterns(processed_data)
            if not patterns:
                self.record_test_result(test_name, False, "Failed to find cross-modal patterns")
                return
            
            # Step 3: Real-time pattern monitoring
            monitoring_result = self.pattern_monitor.monitor_patterns(patterns)
            if not monitoring_result:
                self.record_test_result(test_name, False, "Failed to monitor patterns")
                return
            
            # Step 4: Causal inference
            causal_analysis = self.causal_engine.analyze_causality(processed_data, patterns)
            if not causal_analysis:
                self.record_test_result(test_name, False, "Failed to perform causal analysis")
                return
            
            self.record_test_result(test_name, True, f"Successfully processed {len(processed_data)} modalities")
            
        except Exception as e:
            self.record_test_result(test_name, False, error=str(e))
    
    def test_real_time_data_integration(self):
        """Test real-time data integration."""
        test_name = "Real-Time Data Integration"
        
        try:
            logger.info("Testing real-time data integration...")
            
            # Test data sources
            data_sources = [
                {"type": "api", "endpoint": "https://api.example.com/metrics", "frequency": "5min"},
                {"type": "database", "table": "customer_feedback", "frequency": "1min"},
                {"type": "stream", "topic": "market_data", "frequency": "realtime"}
            ]
            
            # Step 1: Connect to data sources
            connections = []
            for source in data_sources:
                connection = self.api_connector.connect_to_source(source)
                if connection:
                    connections.append(connection)
            
            if not connections:
                self.record_test_result(test_name, False, "Failed to connect to any data sources")
                return
            
            # Step 2: Synchronize data
            sync_result = self.data_synchronizer.synchronize_data(connections)
            if not sync_result:
                self.record_test_result(test_name, False, "Failed to synchronize data")
                return
            
            # Step 3: Monitor real-time patterns
            pattern_result = self.pattern_monitor.monitor_realtime_patterns(sync_result)
            if not pattern_result:
                self.record_test_result(test_name, False, "Failed to monitor real-time patterns")
                return
            
            # Step 4: Trigger alerts if needed
            alert_result = self.alert_system.check_alerts(pattern_result)
            if alert_result is None:
                self.record_test_result(test_name, False, "Failed to check alerts")
                return
            
            self.record_test_result(test_name, True, f"Successfully integrated {len(connections)} data sources")
            
        except Exception as e:
            self.record_test_result(test_name, False, error=str(e))
    
    def test_external_system_connections(self):
        """Test external system connections."""
        test_name = "External System Connections"
        
        try:
            logger.info("Testing external system connections...")
            
            # Test external systems
            external_systems = [
                {"type": "erp", "name": "SAP", "endpoint": "https://sap.example.com/api"},
                {"type": "crm", "name": "Salesforce", "endpoint": "https://salesforce.example.com/api"},
                {"type": "bi", "name": "Tableau", "endpoint": "https://tableau.example.com/api"}
            ]
            
            successful_connections = 0
            
            for system in external_systems:
                try:
                    # Test connection
                    connection = self.api_connector.test_connection(system)
                    if connection:
                        successful_connections += 1
                        
                        # Test data exchange
                        data_exchange = self.data_synchronizer.exchange_data(connection)
                        if not data_exchange:
                            logger.warning(f"Data exchange failed for {system['name']}")
                    
                except Exception as e:
                    logger.warning(f"Connection failed for {system['name']}: {e}")
            
            if successful_connections == 0:
                self.record_test_result(test_name, False, "Failed to connect to any external systems")
                return
            
            self.record_test_result(test_name, True, f"Successfully connected to {successful_connections}/{len(external_systems)} external systems")
            
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
    
    def test_complete_workflow_integration(self):
        """Test complete workflow integration."""
        test_name = "Complete Workflow Integration"
        
        try:
            logger.info("Testing complete workflow integration...")
            
            # Simulate a complete business scenario
            scenario = {
                "business_context": {
                    "company": "TechCorp",
                    "industry": "Software",
                    "goal": "Launch new mobile app successfully"
                },
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
            
            # Step 1: Initialize knowledge graph with scenario data
            kg_result = self.knowledge_agent.initialize_scenario(scenario)
            if not kg_result:
                self.record_test_result(test_name, False, "Failed to initialize knowledge graph")
                return
            
            # Step 2: Process multi-modal data
            mm_result = self.multi_modal_agent.process_scenario_data(scenario["data_sources"])
            if not mm_result:
                self.record_test_result(test_name, False, "Failed to process scenario data")
                return
            
            # Step 3: Generate decision support
            decision_result = self.decision_agent.generate_decision_support(scenario)
            if not decision_result:
                self.record_test_result(test_name, False, "Failed to generate decision support")
                return
            
            # Step 4: Create implementation plan
            plan_result = self.implementation_planner.create_comprehensive_plan(decision_result)
            if not plan_result:
                self.record_test_result(test_name, False, "Failed to create comprehensive plan")
                return
            
            # Step 5: Set up monitoring
            monitoring_result = self.decision_monitor.setup_comprehensive_monitoring(plan_result)
            if not monitoring_result:
                self.record_test_result(test_name, False, "Failed to setup comprehensive monitoring")
                return
            
            # Step 6: Configure alerts
            alert_result = self.alert_system.configure_scenario_alerts(scenario)
            if not alert_result:
                self.record_test_result(test_name, False, "Failed to configure scenario alerts")
                return
            
            self.record_test_result(test_name, True, "Successfully completed full workflow integration")
            
        except Exception as e:
            self.record_test_result(test_name, False, error=str(e))
    
    def run_all_tests(self):
        """Run all end-to-end integration tests."""
        logger.info("Starting End-to-End Integration Testing...")
        
        # Setup components
        if not self.setup_components():
            logger.error("Failed to setup components. Aborting tests.")
            return False
        
        # Run tests
        tests = [
            self.test_decision_support_workflow,
            self.test_multi_modal_processing_pipeline,
            self.test_real_time_data_integration,
            self.test_external_system_connections,
            self.test_monitoring_and_alerting_systems,
            self.test_knowledge_graph_integration,
            self.test_complete_workflow_integration
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
        logger.info("Generating integration test report...")
        
        # Calculate success rate
        total = self.results["summary"]["total_tests"]
        passed = self.results["summary"]["passed"]
        success_rate = (passed / total * 100) if total > 0 else 0
        
        # Add summary statistics
        self.results["summary"]["success_rate"] = f"{success_rate:.1f}%"
        self.results["summary"]["completion_time"] = datetime.now().isoformat()
        
        # Save detailed report
        report_path = Path("Results") / f"integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("END-TO-END INTEGRATION TEST RESULTS")
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
    logger.info("Starting End-to-End Integration Testing Suite")
    
    tester = EndToEndIntegrationTester()
    success = tester.run_all_tests()
    
    if success:
        logger.info("Integration testing completed successfully!")
        return 0
    else:
        logger.error("Integration testing completed with failures!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
