"""
Real-Time Monitoring Agent

This agent coordinates all real-time monitoring components including pattern
monitoring, alerts, performance dashboard, and data stream processing.
"""

import asyncio
import logging
import uuid
from typing import Dict, Optional, Any, List
from datetime import datetime

from .base_agent import StrandsBaseAgent
from ..core.models import DataType, AnalysisRequest, AnalysisResult, SentimentResult, SentimentLabel
from ..core.real_time.pattern_monitor import RealTimePatternMonitor, MonitoringConfig
from ..core.real_time.alert_system import AlertSystem, AlertConfig, AlertRule
from ..core.real_time.performance_dashboard import PerformanceDashboard, DashboardConfig
from ..core.real_time.stream_processor import DataStreamProcessor, StreamConfig

logger = logging.getLogger(__name__)


class RealTimeMonitoringAgent(StrandsBaseAgent):
    """
    Agent for real-time monitoring capabilities including pattern detection,
    alerting, performance monitoring, and data stream processing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the real-time monitoring agent
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(
            agent_id=f"RealTimeMonitoringAgent_{str(uuid.uuid4())[:8]}"
        )
        
        # Store config
        self.config = config or {}
        
        # Set agent type
        self.agent_type = "real_time_monitoring"
        
        # Initialize real-time monitoring components
        self.pattern_monitor = RealTimePatternMonitor(
            MonitoringConfig(**self.config.get('pattern_monitor', {}))
        )
        self.alert_system = AlertSystem(
            AlertConfig(**self.config.get('alert_system', {}))
        )
        self.performance_dashboard = PerformanceDashboard(
            DashboardConfig(**self.config.get('performance_dashboard', {}))
        )
        self.stream_processor = DataStreamProcessor(
            StreamConfig(**self.config.get('stream_processor', {}))
        )
        
        # Supported data types
        self.supported_data_types = [
            DataType.NUMERICAL,
            DataType.TIME_SERIES,
            DataType.TEXT
        ]
        
        # Set up component integration
        self._setup_integration()
        
        logger.info(f"RealTimeMonitoringAgent {self.agent_id} initialized")
    
    def _setup_integration(self):
        """Set up integration between monitoring components"""
        # Connect pattern monitor to alert system
        self.pattern_monitor.add_callback(self._on_pattern_detected)
        
        # Connect performance dashboard to alert system
        self.performance_dashboard.add_alert_callback(self._on_dashboard_alert)
        
        # Connect stream processor to pattern monitor
        self.stream_processor.register_consumer(
            "pattern_input", 
            self._on_stream_data
        )
        
        logger.info("Real-time monitoring components integrated")
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the given request."""
        try:
            # Check data type
            if request.data_type not in self.supported_data_types:
                return False
            
            # Check if content is provided
            if not request.content:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking if can process: {str(e)}")
            return False
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Process a real-time monitoring request
        
        Args:
            request: Analysis request with data and parameters
            
        Returns:
            AnalysisResult with monitoring results
        """
        try:
            logger.info(f"Processing real-time monitoring request: {request.id}")
            
            # Validate request
            if not self._validate_request(request):
                return AnalysisResult(
                    id=str(uuid.uuid4()),
                    request_id=request.id,
                    data_type=request.data_type,
                    sentiment=SentimentResult(
                        label=SentimentLabel.UNCERTAIN,
                        confidence=0.0,
                        reasoning="Invalid request: Unsupported data type or missing data"
                    ),
                    processing_time=0.0,
                    status="failed"
                )
            
            # Extract data and parameters
            data = self._extract_data(request)
            if data is None:
                return AnalysisResult(
                    id=str(uuid.uuid4()),
                    request_id=request.id,
                    data_type=request.data_type,
                    sentiment=SentimentResult(
                        label=SentimentLabel.UNCERTAIN,
                        confidence=0.0,
                        reasoning="Failed to extract data from request"
                    ),
                    processing_time=0.0,
                    status="failed"
                )
            
            # Determine monitoring type
            monitoring_type = request.metadata.get('monitoring_type', 'pattern')
            
            # Perform monitoring based on type
            if monitoring_type == 'pattern':
                result = await self._perform_pattern_monitoring(data, request.metadata)
            elif monitoring_type == 'performance':
                result = await self._perform_performance_monitoring(data, request.metadata)
            elif monitoring_type == 'stream':
                result = await self._perform_stream_processing(data, request.metadata)
            elif monitoring_type == 'alert':
                result = await self._perform_alert_management(data, request.metadata)
            else:
                return AnalysisResult(
                    id=str(uuid.uuid4()),
                    request_id=request.id,
                    data_type=request.data_type,
                    sentiment=SentimentResult(
                        label=SentimentLabel.UNCERTAIN,
                        confidence=0.0,
                        reasoning=f"Unsupported monitoring type: {monitoring_type}"
                    ),
                    processing_time=0.0,
                    status="failed"
                )
            
            # Create analysis result
            return AnalysisResult(
                id=str(uuid.uuid4()),
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label=SentimentLabel.NEUTRAL,
                    confidence=1.0,
                    reasoning=f"Real-time monitoring completed: {monitoring_type}"
                ),
                processing_time=0.0,
                status="completed",
                metadata={
                    'monitoring_type': monitoring_type,
                    'data_points': len(data) if isinstance(data, list) else 1,
                    'timestamp': datetime.now().isoformat(),
                    'agent_type': self.agent_type,
                    'monitoring_result': result
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing real-time monitoring request: {str(e)}")
            return AnalysisResult(
                id=str(uuid.uuid4()),
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label=SentimentLabel.UNCERTAIN,
                    confidence=0.0,
                    reasoning=f"Processing error: {str(e)}"
                ),
                processing_time=0.0,
                status="failed"
            )
    
    async def _perform_pattern_monitoring(self, data: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform pattern monitoring analysis"""
        try:
            # Start monitoring if not already running
            if not self.pattern_monitor.is_monitoring:
                await self.pattern_monitor.start_monitoring()
            
            # Add data points to monitor
            if isinstance(data, list):
                for i, value in enumerate(data):
                    self.pattern_monitor.add_data_point(
                        value=float(value),
                        timestamp=datetime.now()
                    )
            else:
                self.pattern_monitor.add_data_point(
                    value=float(data),
                    timestamp=datetime.now()
                )
            
            # Get monitoring statistics
            stats = self.pattern_monitor.get_statistics()
            recent_patterns = self.pattern_monitor.get_recent_patterns(limit=10)
            
            return {
                'monitoring_status': 'active',
                'statistics': stats,
                'recent_patterns': [
                    {
                        'pattern_id': p.pattern_id,
                        'pattern_type': p.pattern_type,
                        'confidence': p.confidence,
                        'severity': p.severity,
                        'timestamp': p.timestamp.isoformat()
                    }
                    for p in recent_patterns
                ],
                'analysis_type': 'pattern_monitoring'
            }
            
        except Exception as e:
            logger.error(f"Error in pattern monitoring: {str(e)}")
            raise
    
    async def _perform_performance_monitoring(self, data: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform performance monitoring analysis"""
        try:
            # Start dashboard if not already running
            if not self.performance_dashboard.is_monitoring:
                await self.performance_dashboard.start_monitoring()
            
            # Add custom metrics if provided
            if isinstance(data, dict):
                for metric_name, metric_value in data.items():
                    self.performance_dashboard.add_metric(
                        name=metric_name,
                        value=float(metric_value),
                        category="custom"
                    )
            
            # Get performance report
            report = self.performance_dashboard.get_performance_report()
            
            return {
                'monitoring_status': 'active',
                'performance_report': report,
                'analysis_type': 'performance_monitoring'
            }
            
        except Exception as e:
            logger.error(f"Error in performance monitoring: {str(e)}")
            raise
    
    async def _perform_stream_processing(self, data: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform data stream processing"""
        try:
            # Start stream processing if not already running
            if not self.stream_processor.is_processing:
                await self.stream_processor.start_processing()
            
            # Add data to stream
            if isinstance(data, list):
                for value in data:
                    self.stream_processor.add_data_point(
                        value=value,
                        source=parameters.get('source', 'default')
                    )
            else:
                self.stream_processor.add_data_point(
                    value=data,
                    source=parameters.get('source', 'default')
                )
            
            # Get processing statistics
            stats = self.stream_processor.get_processing_stats()
            
            return {
                'processing_status': 'active',
                'statistics': stats,
                'analysis_type': 'stream_processing'
            }
            
        except Exception as e:
            logger.error(f"Error in stream processing: {str(e)}")
            raise
    
    async def _perform_alert_management(self, data: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform alert management operations"""
        try:
            operation = parameters.get('operation', 'get_alerts')
            
            if operation == 'add_rule':
                # Add new alert rule
                rule = AlertRule(
                    rule_id=parameters.get('rule_id', f"rule_{uuid.uuid4().hex[:8]}"),
                    pattern_type=parameters.get('pattern_type', 'anomaly'),
                    severity_threshold=parameters.get('severity_threshold', 'warning'),
                    conditions=parameters.get('conditions', {}),
                    actions=parameters.get('actions', ['log'])
                )
                self.alert_system.add_rule(rule)
                
                return {
                    'operation': 'add_rule',
                    'rule_id': rule.rule_id,
                    'status': 'success'
                }
            
            elif operation == 'get_alerts':
                # Get alerts with optional filtering
                alerts = self.alert_system.get_alerts(
                    severity=parameters.get('severity'),
                    pattern_type=parameters.get('pattern_type'),
                    acknowledged=parameters.get('acknowledged'),
                    limit=parameters.get('limit', 50)
                )
                
                return {
                    'operation': 'get_alerts',
                    'alerts': [
                        {
                            'alert_id': a.alert_id,
                            'pattern_type': a.pattern_type,
                            'severity': a.severity,
                            'message': a.message,
                            'timestamp': a.timestamp.isoformat(),
                            'acknowledged': a.acknowledged
                        }
                        for a in alerts
                    ],
                    'total_count': len(alerts)
                }
            
            elif operation == 'acknowledge':
                # Acknowledge an alert
                alert_id = parameters.get('alert_id')
                acknowledged_by = parameters.get('acknowledged_by', 'system')
                
                if alert_id:
                    self.alert_system.acknowledge_alert(alert_id, acknowledged_by)
                    return {
                        'operation': 'acknowledge',
                        'alert_id': alert_id,
                        'status': 'acknowledged'
                    }
                else:
                    raise ValueError("alert_id is required for acknowledge operation")
            
            else:
                raise ValueError(f"Unsupported alert operation: {operation}")
            
        except Exception as e:
            logger.error(f"Error in alert management: {str(e)}")
            raise
    
    def _on_pattern_detected(self, pattern_event):
        """Callback for pattern detection events"""
        try:
            # Process pattern event through alert system
            asyncio.create_task(self.alert_system.process_pattern_event(pattern_event))
            
            # Add to performance dashboard
            self.performance_dashboard.add_metric(
                name=f"pattern_{pattern_event.pattern_type}",
                value=pattern_event.confidence,
                unit="confidence",
                category="patterns",
                metadata={
                    'pattern_id': pattern_event.pattern_id,
                    'severity': pattern_event.severity
                }
            )
            
        except Exception as e:
            logger.error(f"Error handling pattern event: {str(e)}")
    
    def _on_dashboard_alert(self, alert_message: str, metric):
        """Callback for dashboard alert events"""
        try:
            logger.warning(f"Dashboard Alert: {alert_message}")
            
            # Could route to external alerting systems here
            # For now, just log the alert
            
        except Exception as e:
            logger.error(f"Error handling dashboard alert: {str(e)}")
    
    def _on_stream_data(self, data):
        """Callback for stream data events"""
        try:
            # Route stream data to pattern monitor if it's numeric
            if isinstance(data, (int, float)):
                self.pattern_monitor.add_data_point(value=float(data))
            
            # Add to performance dashboard
            if hasattr(data, 'value') and hasattr(data, 'timestamp'):
                self.performance_dashboard.add_metric(
                    name="stream_data",
                    value=float(data.value),
                    unit="value",
                    category="stream"
                )
            
        except Exception as e:
            logger.error(f"Error handling stream data: {str(e)}")
    
    def _validate_request(self, request: AnalysisRequest) -> bool:
        """Validate analysis request"""
        try:
            # Check data type
            if request.data_type not in self.supported_data_types:
                return False
            
            # Check if content is provided
            if not request.content:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating request: {str(e)}")
            return False
    
    def _extract_data(self, request: AnalysisRequest) -> Optional[Any]:
        """Extract and convert data from request"""
        try:
            # Extract data from content
            data = request.content.get('data')
            if data is None:
                logger.error("No data found in request content")
                return None
            
            return data
            
        except Exception as e:
            logger.error(f"Error extracting data: {str(e)}")
            return None
    
    async def start_monitoring(self):
        """Start all monitoring components"""
        try:
            await self.pattern_monitor.start_monitoring()
            await self.performance_dashboard.start_monitoring()
            await self.stream_processor.start_processing()
            logger.info("All real-time monitoring components started")
        except Exception as e:
            logger.error(f"Error starting monitoring: {str(e)}")
    
    async def stop_monitoring(self):
        """Stop all monitoring components"""
        try:
            await self.pattern_monitor.stop_monitoring()
            await self.performance_dashboard.stop_monitoring()
            await self.stream_processor.stop_processing()
            logger.info("All real-time monitoring components stopped")
        except Exception as e:
            logger.error(f"Error stopping monitoring: {str(e)}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get overall monitoring status"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'pattern_monitor': {
                'is_monitoring': self.pattern_monitor.is_monitoring,
                'statistics': self.pattern_monitor.get_statistics()
            },
            'performance_dashboard': {
                'is_monitoring': self.performance_dashboard.is_monitoring,
                'summary': self.performance_dashboard.get_dashboard_summary()
            },
            'stream_processor': {
                'is_processing': self.stream_processor.is_processing,
                'statistics': self.stream_processor.get_processing_stats()
            },
            'alert_system': {
                'statistics': self.alert_system.get_statistics()
            }
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities"""
        return {
            'agent_type': self.agent_type,
            'agent_id': self.agent_id,
            'supported_data_types': [dt.value for dt in self.supported_data_types],
            'monitoring_types': [
                'pattern',
                'performance', 
                'stream',
                'alert'
            ],
            'pattern_types': ['anomaly', 'trend', 'seasonal', 'spike'],
            'alert_channels': ['email', 'webhook', 'slack', 'log'],
            'stream_operations': ['filter', 'process', 'aggregate']
        }
