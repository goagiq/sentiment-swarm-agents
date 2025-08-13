"""
Comprehensive Test Suite for Risk Management System

This test suite validates all components of the risk management system including
risk identification, quantification, mitigation planning, and monitoring.
"""

import unittest
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.risk_management.risk_identifier import RiskIdentifier, Risk
from core.risk_management.risk_quantifier import RiskQuantifier, RiskImpact
from core.risk_management.mitigation_planner import MitigationPlanner, MitigationPlan, MitigationAction
from core.risk_management.risk_monitor import RiskMonitor, RiskAlert, RiskTrend
from agents.risk_assessment_agent import RiskAssessmentAgent


class TestRiskIdentifier(unittest.TestCase):
    """Test cases for RiskIdentifier component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.risk_identifier = RiskIdentifier()
    
    def test_identify_data_quality_risks(self):
        """Test data quality risk identification."""
        data_metrics = {
            'missing_rate': 0.15,  # Above threshold
            'duplicate_rate': 0.08,  # Above threshold
            'inconsistent_data': 0.12
        }
        
        risks = self.risk_identifier.identify_data_quality_risks(data_metrics)
        
        self.assertIsInstance(risks, list)
        self.assertGreater(len(risks), 0)
        
        for risk in risks:
            self.assertIsInstance(risk, Risk)
            self.assertEqual(risk.category, 'data_quality')
            self.assertIn(risk.severity, ['low', 'medium', 'high', 'critical'])
    
    def test_identify_performance_risks(self):
        """Test performance risk identification."""
        performance_metrics = {
            'avg_response_time': 3.5,  # Above threshold
            'error_rate': 0.08,  # Above threshold
            'resource_usage': 0.85
        }
        
        risks = self.risk_identifier.identify_performance_risks(performance_metrics)
        
        self.assertIsInstance(risks, list)
        self.assertGreater(len(risks), 0)
        
        for risk in risks:
            self.assertIsInstance(risk, Risk)
            self.assertEqual(risk.category, 'performance')
    
    def test_identify_security_risks(self):
        """Test security risk identification."""
        security_metrics = {
            'failed_logins': 8,  # Above threshold
            'suspicious_activity_count': 5,  # Above threshold
            'data_access_patterns': 0.05
        }
        
        risks = self.risk_identifier.identify_security_risks(security_metrics)
        
        self.assertIsInstance(risks, list)
        self.assertGreater(len(risks), 0)
        
        for risk in risks:
            self.assertIsInstance(risk, Risk)
            self.assertEqual(risk.category, 'security')
    
    def test_identify_business_risks(self):
        """Test business risk identification."""
        business_metrics = {
            'revenue_change': -0.15,  # Below threshold
            'churn_rate': 0.08,  # Above threshold
            'operational_efficiency': 0.12
        }
        
        risks = self.risk_identifier.identify_business_risks(business_metrics)
        
        self.assertIsInstance(risks, list)
        self.assertGreater(len(risks), 0)
        
        for risk in risks:
            self.assertIsInstance(risk, Risk)
            self.assertEqual(risk.category, 'business')
    
    def test_identify_all_risks(self):
        """Test comprehensive risk identification."""
        data_metrics = {'missing_rate': 0.15}
        performance_metrics = {'avg_response_time': 3.5}
        security_metrics = {'failed_logins': 8}
        business_metrics = {'revenue_change': -0.15}
        
        risks = self.risk_identifier.identify_all_risks(
            data_metrics=data_metrics,
            performance_metrics=performance_metrics,
            security_metrics=security_metrics,
            business_metrics=business_metrics
        )
        
        self.assertIsInstance(risks, list)
        self.assertGreater(len(risks), 0)
        
        categories = [risk.category for risk in risks]
        self.assertIn('data_quality', categories)
        self.assertIn('performance', categories)
        self.assertIn('security', categories)
        self.assertIn('business', categories)
    
    def test_get_risk_summary(self):
        """Test risk summary generation."""
        data_metrics = {'missing_rate': 0.15}
        risks = self.risk_identifier.identify_data_quality_risks(data_metrics)
        
        summary = self.risk_identifier.get_risk_summary(risks)
        
        self.assertIsInstance(summary, dict)
        self.assertIn('total_risks', summary)
        self.assertIn('risk_distribution', summary)
        self.assertIn('category_distribution', summary)
        self.assertEqual(summary['total_risks'], len(risks))


class TestRiskQuantifier(unittest.TestCase):
    """Test cases for RiskQuantifier component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.risk_quantifier = RiskQuantifier()
        
        # Create a test risk
        self.test_risk = Risk(
            id="test_risk_001",
            category="data_quality",
            severity="high",
            probability=0.7,
            impact=0.8,
            description="Test data quality risk",
            source="test",
            detected_at=datetime.now(),
            affected_components=["data_pipeline"],
            indicators=["missing_data"],
            confidence=0.8
        )
    
    def test_calculate_financial_impact(self):
        """Test financial impact calculation."""
        business_context = {
            'data_incidents_per_month': 3,
            'affected_team_size': 15,
            'avg_hourly_rate': 60
        }
        
        financial_impact = self.risk_quantifier.calculate_financial_impact(
            self.test_risk, business_context
        )
        
        self.assertIsInstance(financial_impact, float)
        self.assertGreater(financial_impact, 0)
    
    def test_calculate_operational_impact(self):
        """Test operational impact calculation."""
        operational_context = {
            'business_critical_hours': True,
            'peak_usage_period': False
        }
        
        operational_impact = self.risk_quantifier.calculate_operational_impact(
            self.test_risk, operational_context
        )
        
        self.assertIsInstance(operational_impact, float)
        self.assertGreaterEqual(operational_impact, 0)
        self.assertLessEqual(operational_impact, 1)
    
    def test_calculate_strategic_impact(self):
        """Test strategic impact calculation."""
        strategic_context = {
            'core_business_function': True,
            'competitive_advantage': False
        }
        
        strategic_impact = self.risk_quantifier.calculate_strategic_impact(
            self.test_risk, strategic_context
        )
        
        self.assertIsInstance(strategic_impact, float)
        self.assertGreaterEqual(strategic_impact, 0)
        self.assertLessEqual(strategic_impact, 1)
    
    def test_quantify_risk_impact(self):
        """Test comprehensive risk impact quantification."""
        business_context = {'data_incidents_per_month': 2}
        operational_context = {'business_critical_hours': True}
        strategic_context = {'core_business_function': True}
        reputation_context = {'public_facing': False}
        compliance_context = {'gdpr_applicable': True}
        
        impact = self.risk_quantifier.quantify_risk_impact(
            self.test_risk,
            business_context=business_context,
            operational_context=operational_context,
            strategic_context=strategic_context,
            reputation_context=reputation_context,
            compliance_context=compliance_context
        )
        
        self.assertIsInstance(impact, RiskImpact)
        self.assertEqual(impact.risk_id, self.test_risk.id)
        self.assertGreater(impact.financial_impact, 0)
        self.assertGreaterEqual(impact.total_impact_score, 0)
        self.assertLessEqual(impact.total_impact_score, 1)
    
    def test_quantify_multiple_risks(self):
        """Test quantification of multiple risks."""
        risks = [self.test_risk]
        context = {
            'business': {'data_incidents_per_month': 2},
            'operational': {'business_critical_hours': True}
        }
        
        impacts = self.risk_quantifier.quantify_multiple_risks(risks, context)
        
        self.assertIsInstance(impacts, list)
        self.assertEqual(len(impacts), len(risks))
        
        for impact in impacts:
            self.assertIsInstance(impact, RiskImpact)
    
    def test_get_impact_summary(self):
        """Test impact summary generation."""
        risks = [self.test_risk]
        context = {'business': {'data_incidents_per_month': 2}}
        impacts = self.risk_quantifier.quantify_multiple_risks(risks, context)
        
        summary = self.risk_quantifier.get_impact_summary(impacts)
        
        self.assertIsInstance(summary, dict)
        self.assertIn('total_financial_impact', summary)
        self.assertIn('avg_impact_score', summary)
        self.assertIn('avg_confidence', summary)


class TestMitigationPlanner(unittest.TestCase):
    """Test cases for MitigationPlanner component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mitigation_planner = MitigationPlanner()
        
        # Create test risk and impact
        self.test_risk = Risk(
            id="test_risk_002",
            category="security",
            severity="critical",
            probability=0.8,
            impact=0.9,
            description="Test security risk",
            source="test",
            detected_at=datetime.now(),
            affected_components=["authentication"],
            indicators=["failed_logins"],
            confidence=0.9
        )
        
        self.test_impact = RiskImpact(
            risk_id="test_risk_002",
            financial_impact=50000,
            operational_impact=0.8,
            strategic_impact=0.7,
            reputation_impact=0.9,
            compliance_impact=0.8,
            total_impact_score=0.8,
            impact_breakdown={},
            confidence_level=0.8,
            assessment_date=datetime.now()
        )
    
    def test_create_mitigation_actions(self):
        """Test mitigation action creation."""
        actions = self.mitigation_planner.create_mitigation_actions(
            self.test_risk, self.test_impact
        )
        
        self.assertIsInstance(actions, list)
        self.assertGreater(len(actions), 0)
        
        for action in actions:
            self.assertIsInstance(action, MitigationAction)
            self.assertEqual(action.risk_id, self.test_risk.id)
            self.assertIn(action.action_type, ['preventive', 'detective', 'corrective'])
            self.assertIn(action.priority, ['low', 'medium', 'high', 'critical'])
    
    def test_create_mitigation_plan(self):
        """Test mitigation plan creation."""
        plan = self.mitigation_planner.create_mitigation_plan(
            self.test_risk, self.test_impact
        )
        
        self.assertIsInstance(plan, MitigationPlan)
        self.assertEqual(plan.risk_id, self.test_risk.id)
        self.assertEqual(plan.current_risk_level, self.test_risk.severity)
        self.assertGreater(len(plan.actions), 0)
        self.assertGreater(plan.total_cost, 0)
        self.assertGreater(plan.total_duration, 0)
    
    def test_optimize_mitigation_plan(self):
        """Test mitigation plan optimization."""
        plan = self.mitigation_planner.create_mitigation_plan(
            self.test_risk, self.test_impact
        )
        
        # Optimize with budget constraint
        optimized_plan = self.mitigation_planner.optimize_mitigation_plan(
            plan, budget_constraint=50000
        )
        
        self.assertIsInstance(optimized_plan, MitigationPlan)
        self.assertLessEqual(optimized_plan.total_cost, 50000)
    
    def test_get_mitigation_summary(self):
        """Test mitigation summary generation."""
        plan = self.mitigation_planner.create_mitigation_plan(
            self.test_risk, self.test_impact
        )
        
        summary = self.mitigation_planner.get_mitigation_summary([plan])
        
        self.assertIsInstance(summary, dict)
        self.assertIn('total_plans', summary)
        self.assertIn('total_cost', summary)
        self.assertIn('avg_risk_reduction', summary)


class TestRiskMonitor(unittest.TestCase):
    """Test cases for RiskMonitor component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.risk_monitor = RiskMonitor()
        
        # Create test risk
        self.test_risk = Risk(
            id="test_risk_003",
            category="performance",
            severity="medium",
            probability=0.6,
            impact=0.7,
            description="Test performance risk",
            source="test",
            detected_at=datetime.now(),
            affected_components=["api"],
            indicators=["response_time"],
            confidence=0.7
        )
    
    def test_add_risk_to_history(self):
        """Test adding risk to monitoring history."""
        self.risk_monitor.add_risk_to_history(self.test_risk)
        
        self.assertIn(self.test_risk.id, self.risk_monitor.risk_history)
        self.assertEqual(len(self.risk_monitor.risk_history[self.test_risk.id]), 1)
    
    def test_calculate_risk_score(self):
        """Test risk score calculation."""
        score = self.risk_monitor.calculate_risk_score(self.test_risk)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
    
    def test_check_alert_conditions(self):
        """Test alert condition checking."""
        alerts = self.risk_monitor.check_alert_conditions(self.test_risk)
        
        self.assertIsInstance(alerts, list)
        
        for alert in alerts:
            self.assertIsInstance(alert, RiskAlert)
            self.assertEqual(alert.risk_id, self.test_risk.id)
    
    def test_process_risk_update(self):
        """Test risk update processing."""
        alerts = self.risk_monitor.process_risk_update(self.test_risk)
        
        self.assertIsInstance(alerts, list)
        self.assertIn(self.test_risk.id, self.risk_monitor.risk_history)
    
    def test_get_active_alerts(self):
        """Test active alerts retrieval."""
        # First add a risk to generate alerts
        self.risk_monitor.process_risk_update(self.test_risk)
        
        alerts = self.risk_monitor.get_active_alerts()
        
        self.assertIsInstance(alerts, list)
    
    def test_get_risk_summary(self):
        """Test risk summary generation."""
        # Add some risks first
        self.risk_monitor.process_risk_update(self.test_risk)
        
        summary = self.risk_monitor.get_risk_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('total_risks_monitored', summary)
        self.assertIn('active_alerts', summary)
        self.assertIn('monitoring_active', summary)


class TestRiskAssessmentAgent(unittest.TestCase):
    """Test cases for RiskAssessmentAgent component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = RiskAssessmentAgent()
    
    def test_assess_risks(self):
        """Test comprehensive risk assessment."""
        data_metrics = {'missing_rate': 0.15}
        performance_metrics = {'avg_response_time': 3.5}
        security_metrics = {'failed_logins': 8}
        business_metrics = {'revenue_change': -0.15}
        
        result = self.agent.assess_risks(
            data_metrics=data_metrics,
            performance_metrics=performance_metrics,
            security_metrics=security_metrics,
            business_metrics=business_metrics
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
        self.assertIn('risks', result)
        self.assertIn('impacts', result)
        self.assertIn('mitigation_plans', result)
        self.assertIn('alerts', result)
        self.assertIn('summaries', result)
        self.assertIn('recommendations', result)
    
    def test_get_active_alerts(self):
        """Test active alerts retrieval."""
        result = self.agent.get_active_alerts()
        
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
        self.assertIn('alerts', result)
        self.assertIn('total_alerts', result)
    
    def test_get_risk_trends(self):
        """Test risk trends retrieval."""
        result = self.agent.get_risk_trends(period_days=7)
        
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
        self.assertIn('trends', result)
        self.assertIn('total_trends', result)
    
    def test_optimize_mitigation_plan(self):
        """Test mitigation plan optimization."""
        result = self.agent.optimize_mitigation_plan(
            "test_plan_id",
            budget_constraint=50000,
            timeline_constraint=30
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
        self.assertIn('message', result)
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'agent'):
            self.agent.shutdown()


class TestRiskManagementIntegration(unittest.TestCase):
    """Integration tests for the complete risk management system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = RiskAssessmentAgent()
    
    def test_end_to_end_risk_assessment(self):
        """Test complete end-to-end risk assessment workflow."""
        # Step 1: Provide comprehensive metrics
        data_metrics = {
            'missing_rate': 0.15,
            'duplicate_rate': 0.08,
            'inconsistent_data': 0.12
        }
        
        performance_metrics = {
            'avg_response_time': 3.5,
            'error_rate': 0.08,
            'resource_usage': 0.85
        }
        
        security_metrics = {
            'failed_logins': 8,
            'suspicious_activity_count': 5,
            'data_access_patterns': 0.05
        }
        
        business_metrics = {
            'revenue_change': -0.15,
            'churn_rate': 0.08,
            'operational_efficiency': 0.12
        }
        
        context = {
            'business': {
                'data_incidents_per_month': 3,
                'affected_team_size': 15,
                'avg_hourly_rate': 60,
                'monthly_revenue': 100000
            },
            'operational': {
                'business_critical_hours': True,
                'peak_usage_period': False
            },
            'strategic': {
                'core_business_function': True,
                'competitive_advantage': False
            },
            'reputation': {
                'public_facing': True,
                'media_attention': False
            },
            'compliance': {
                'gdpr_applicable': True,
                'sox_compliance': False
            }
        }
        
        # Step 2: Perform comprehensive assessment
        result = self.agent.assess_risks(
            data_metrics=data_metrics,
            performance_metrics=performance_metrics,
            security_metrics=security_metrics,
            business_metrics=business_metrics,
            context=context
        )
        
        # Step 3: Validate results
        self.assertEqual(result['status'], 'success')
        self.assertGreater(len(result['risks']), 0)
        self.assertGreater(len(result['impacts']), 0)
        self.assertGreater(len(result['mitigation_plans']), 0)
        self.assertGreater(len(result['recommendations']), 0)
        
        # Step 4: Validate risk categories
        risk_categories = [risk['category'] for risk in result['risks']]
        expected_categories = ['data_quality', 'performance', 'security', 'business']
        
        for category in expected_categories:
            self.assertIn(category, risk_categories)
        
        # Step 5: Validate impact calculations
        for impact in result['impacts']:
            self.assertGreater(impact['financial_impact'], 0)
            self.assertGreaterEqual(impact['total_impact_score'], 0)
            self.assertLessEqual(impact['total_impact_score'], 1)
        
        # Step 6: Validate mitigation plans
        for plan in result['mitigation_plans']:
            self.assertGreater(len(plan['actions']), 0)
            self.assertGreater(plan['total_cost'], 0)
            self.assertGreater(plan['total_duration'], 0)
        
        # Step 7: Validate summaries
        summaries = result['summaries']
        self.assertIn('risk_summary', summaries)
        self.assertIn('impact_summary', summaries)
        self.assertIn('mitigation_summary', summaries)
        self.assertIn('monitoring_summary', summaries)
    
    def test_risk_monitoring_workflow(self):
        """Test risk monitoring workflow."""
        # Step 1: Perform initial assessment
        data_metrics = {'missing_rate': 0.15}
        performance_metrics = {'avg_response_time': 3.5}
        
        initial_result = self.agent.assess_risks(
            data_metrics=data_metrics,
            performance_metrics=performance_metrics
        )
        
        # Step 2: Check active alerts
        alerts_result = self.agent.get_active_alerts()
        self.assertEqual(alerts_result['status'], 'success')
        
        # Step 3: Check risk trends
        trends_result = self.agent.get_risk_trends(period_days=7)
        self.assertEqual(trends_result['status'], 'success')
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'agent'):
            self.agent.shutdown()


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
