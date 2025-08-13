"""
Fault Detection Agent

Orchestrates all fault detection components including health monitoring,
performance analysis, error prediction, and recovery recommendations.
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

from src.agents.base_agent import StrandsBaseAgent
from src.core.models import AnalysisRequest, AnalysisResult
from src.core.fault_detection import (
    SystemHealthMonitor,
    PerformanceAnalyzer,
    ErrorPredictor,
    RecoveryRecommender
)

logger = logging.getLogger(__name__)


class FaultDetectionAgent(StrandsBaseAgent):
    """Orchestrates comprehensive fault detection and recovery."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        self.config = config or {}
        
        # Initialize base agent
        super().__init__(**kwargs)
        
        self.agent_active = False
        self.agent_thread: Optional[threading.Thread] = None
        
        # Initialize fault detection components
        self.health_monitor = SystemHealthMonitor(
            self.config.get('health_monitor', {})
        )
        self.performance_analyzer = PerformanceAnalyzer(
            self.config.get('performance_analyzer', {})
        )
        self.error_predictor = ErrorPredictor(
            self.config.get('error_predictor', {})
        )
        self.recovery_recommender = RecoveryRecommender(
            self.config.get('recovery_recommender', {})
        )
        
        # Agent state
        self.current_health_status = "unknown"
        self.current_performance_score = 0.0
        self.active_predictions = []
        self.current_recovery_plan = None
        self.last_analysis_time = None
        
        # Configuration
        self.analysis_interval = self.config.get('analysis_interval', 300)  # 5 minutes
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'health_critical': 0.3,
            'performance_critical': 0.4,
            'prediction_confidence': 0.7
        })
        
    def start_agent(self):
        """Start the fault detection agent"""
        if self.agent_active:
            logger.warning("Fault detection agent is already active")
            return
            
        try:
            # Start all components
            self.health_monitor.start_monitoring()
            self.performance_analyzer.start_analysis()
            self.error_predictor.start_prediction()
            
            # Start agent thread
            self.agent_active = True
            self.agent_thread = threading.Thread(
                target=self._agent_loop, daemon=True
            )
            self.agent_thread.start()
            
            logger.info("Fault detection agent started successfully")
            
        except Exception as e:
            logger.error(f"Error starting fault detection agent: {e}")
            self.stop_agent()
            
    def stop_agent(self):
        """Stop the fault detection agent"""
        self.agent_active = False
        
        # Stop all components
        try:
            self.health_monitor.stop_monitoring()
            self.performance_analyzer.stop_analysis()
            self.error_predictor.stop_prediction()
        except Exception as e:
            logger.error(f"Error stopping components: {e}")
            
        # Wait for agent thread
        if self.agent_thread:
            self.agent_thread.join(timeout=10)
            
        logger.info("Fault detection agent stopped")
        
    def _agent_loop(self):
        """Main agent loop for comprehensive fault detection"""
        while self.agent_active:
            try:
                # Perform comprehensive analysis
                self._perform_comprehensive_analysis()
                
                # Check for critical issues
                self._check_critical_issues()
                
                # Generate recovery recommendations if needed
                self._generate_recovery_plan()
                
                # Wait for next analysis cycle
                time.sleep(self.analysis_interval)
                
            except Exception as e:
                logger.error(f"Error in agent loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
                
    def _perform_comprehensive_analysis(self):
        """Perform comprehensive system analysis"""
        try:
            # Get health status
            health_summary = self.health_monitor.get_health_summary()
            self.current_health_status = health_summary.get('overall_status', 'unknown')
            
            # Get performance analysis
            performance_summary = self.performance_analyzer.get_performance_summary()
            self.current_performance_score = performance_summary.get('overall_score', 0.0)
            
            # Get error predictions
            predictions = self.error_predictor.get_predictions()
            self.active_predictions = predictions
            
            # Update last analysis time
            self.last_analysis_time = datetime.now()
            
            logger.debug(f"Analysis completed - Health: {self.current_health_status}, "
                        f"Performance: {self.current_performance_score:.2f}")
                        
        except Exception as e:
            logger.error(f"Error performing comprehensive analysis: {e}")
            
    def _check_critical_issues(self):
        """Check for critical issues requiring immediate attention"""
        try:
            critical_issues = []
            
            # Check health status
            if self.current_health_status == 'critical':
                critical_issues.append({
                    'type': 'health_critical',
                    'severity': 'critical',
                    'description': 'System health is critical'
                })
                
            # Check performance score
            if self.current_performance_score < self.alert_thresholds['performance_critical']:
                critical_issues.append({
                    'type': 'performance_critical',
                    'severity': 'critical',
                    'description': f'Performance score is critically low: {self.current_performance_score:.2f}'
                })
                
            # Check high-confidence predictions
            for prediction in self.active_predictions:
                if (prediction.get('probability', 0) > self.alert_thresholds['prediction_confidence'] and
                    prediction.get('severity') == 'critical'):
                    critical_issues.append({
                        'type': 'prediction_critical',
                        'severity': 'critical',
                        'description': f'Critical error predicted: {prediction.get("error_type")}',
                        'prediction': prediction
                    })
                    
            # Log critical issues
            if critical_issues:
                logger.critical(f"Critical issues detected: {len(critical_issues)}")
                for issue in critical_issues:
                    logger.critical(f"Critical issue: {issue['description']}")
                    
        except Exception as e:
            logger.error(f"Error checking critical issues: {e}")
            
    def _generate_recovery_plan(self):
        """Generate recovery plan if issues are detected"""
        try:
            # Only generate plan if there are issues
            if (self.current_health_status in ['warning', 'critical'] or
                self.current_performance_score < 0.6 or
                any(p.get('probability', 0) > 0.5 for p in self.active_predictions)):
                
                # Get health data for recovery analysis
                health_data = self.health_monitor.get_health_summary()
                
                # Generate recovery plan
                recovery_plan = self.recovery_recommender.analyze_system_issues(health_data)
                self.current_recovery_plan = recovery_plan
                
                logger.info(f"Recovery plan generated - {len(recovery_plan.recommendations)} recommendations")
                
        except Exception as e:
            logger.error(f"Error generating recovery plan: {e}")
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'agent_status': 'active' if self.agent_active else 'inactive',
                'health_status': self.current_health_status,
                'performance_score': self.current_performance_score,
                'active_predictions': len(self.active_predictions),
                'recovery_plan_available': self.current_recovery_plan is not None,
                'last_analysis': self.last_analysis_time.isoformat() if self.last_analysis_time else None
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
            
    def get_detailed_analysis(self) -> Dict[str, Any]:
        """Get detailed analysis from all components"""
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'health_analysis': self.health_monitor.get_health_summary(),
                'performance_analysis': self.performance_analyzer.get_performance_summary(),
                'error_predictions': self.error_predictor.get_prediction_summary(),
                'recovery_plan': self._serialize_recovery_plan(self.current_recovery_plan),
                'bottlenecks': self.performance_analyzer.identify_bottlenecks(),
                'recovery_statistics': self.recovery_recommender.get_recovery_statistics()
            }
        except Exception as e:
            logger.error(f"Error getting detailed analysis: {e}")
            return {'error': str(e)}
            
    def _serialize_recovery_plan(self, plan) -> Optional[Dict[str, Any]]:
        """Serialize recovery plan for JSON output"""
        if plan is None:
            return None
            
        try:
            return {
                'issue_type': plan.issue_type,
                'severity': plan.severity,
                'estimated_total_time': plan.estimated_total_time,
                'overall_success_probability': plan.overall_success_probability,
                'risk_assessment': plan.risk_assessment,
                'recommendations': [
                    {
                        'action': rec.action.value,
                        'priority': rec.priority.value,
                        'description': rec.description,
                        'expected_impact': rec.expected_impact,
                        'estimated_duration': rec.estimated_duration,
                        'success_probability': rec.success_probability,
                        'risk_level': rec.risk_level,
                        'automated': rec.automated
                    }
                    for rec in plan.recommendations
                ]
            }
        except Exception as e:
            logger.error(f"Error serializing recovery plan: {e}")
            return None
            
    def execute_recovery_action(self, action_name: str) -> Dict[str, Any]:
        """Execute a specific recovery action"""
        try:
            if not self.current_recovery_plan:
                return {'success': False, 'error': 'No recovery plan available'}
                
            # Find the recommendation
            recommendation = None
            for rec in self.current_recovery_plan.recommendations:
                if rec.action.value == action_name:
                    recommendation = rec
                    break
                    
            if not recommendation:
                return {'success': False, 'error': f'Action {action_name} not found in current plan'}
                
            # Execute the action
            result = self.recovery_recommender.execute_recovery_action(recommendation)
            
            logger.info(f"Recovery action {action_name} executed: {result.get('success', False)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing recovery action: {e}")
            return {'success': False, 'error': str(e)}
            
    def register_component(self, component_name: str):
        """Register a component for health monitoring"""
        try:
            self.health_monitor.register_component(component_name)
            logger.info(f"Component registered: {component_name}")
        except Exception as e:
            logger.error(f"Error registering component {component_name}: {e}")
            
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get current system alerts"""
        alerts = []
        
        try:
            # Health alerts
            health_summary = self.health_monitor.get_health_summary()
            if 'alerts' in health_summary:
                alerts.extend(health_summary['alerts'])
                
            # Performance alerts
            bottlenecks = self.performance_analyzer.identify_bottlenecks()
            for bottleneck in bottlenecks:
                alerts.append({
                    'type': 'performance_bottleneck',
                    'component': bottleneck['component'],
                    'metric': bottleneck['metric'],
                    'value': bottleneck['value'],
                    'level': bottleneck['level'],
                    'recommendation': bottleneck['recommendation']
                })
                
            # Prediction alerts
            for prediction in self.active_predictions:
                if prediction.get('probability', 0) > 0.6:
                    alerts.append({
                        'type': 'error_prediction',
                        'error_type': prediction.get('error_type'),
                        'probability': prediction.get('probability'),
                        'severity': prediction.get('severity'),
                        'description': prediction.get('description')
                    })
                    
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            
        return alerts
        
    def export_analysis_report(self, filepath: str):
        """Export comprehensive analysis report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'system_status': self.get_system_status(),
                'detailed_analysis': self.get_detailed_analysis(),
                'alerts': self.get_alerts(),
                'recovery_history': self.recovery_recommender.get_recovery_history(24),
                'prediction_accuracy': self.error_predictor.get_prediction_accuracy()
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            logger.info(f"Analysis report exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting analysis report: {e}")
            raise
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the given request."""
        # Fault detection agent can process text, numerical, and time series data
        supported_types = ['text', 'numerical', 'time_series']
        return request.data_type.value in supported_types
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process the analysis request."""
        try:
            # Parse the request content
            if isinstance(request.content, str):
                try:
                    content_data = json.loads(request.content)
                except json.JSONDecodeError:
                    content_data = {"text": request.content}
            else:
                content_data = request.content
            
            # Perform fault detection analysis
            result = self.get_detailed_analysis()
            
            # Create analysis result
            from src.core.models import SentimentResult
            
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="neutral",
                    confidence=0.9,
                    reasoning="Fault detection analysis completed successfully"
                ),
                processing_time=0.0,
                raw_content=str(request.content),
                extracted_text="",
                metadata={"result": result}
            )
            
        except Exception as e:
            logger.error(f"Error processing fault detection request: {e}")
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                error=str(e),
                success=False
            )
    
    async def process_fault_detection(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process fault detection request (legacy method for compatibility)."""
        return self.get_detailed_analysis()
