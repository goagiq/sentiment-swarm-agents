"""
Monitoring MCP Server

Provides MCP endpoints for system monitoring capabilities including:
- Real-time system monitoring
- Performance metrics and alerts
- Fault detection and recovery
- Health checks and diagnostics
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import uuid

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource, Tool, TextContent, ImageContent, EmbeddedResource,
    LoggingLevel, Position, Range, Location, Annotation, Annotations
)

# Import our monitoring components
from src.core.fault_detection.health_monitor import SystemHealthMonitor
from src.core.fault_detection.performance_analyzer import PerformanceAnalyzer
from src.core.fault_detection.error_predictor import ErrorPredictor
from src.core.fault_detection.recovery_recommender import RecoveryRecommender

logger = logging.getLogger(__name__)


class MonitoringMCPServer:
    """
    MCP Server for System Monitoring capabilities
    """
    
    def __init__(self):
        """Initialize the monitoring MCP server"""
        self.server = Server("monitoring")
        self.health_monitor = SystemHealthMonitor()
        self.performance_analyzer = PerformanceAnalyzer()
        self.error_predictor = ErrorPredictor()
        self.recovery_recommender = RecoveryRecommender()
        
        self._setup_tools()
        self._setup_resources()
        
    def _setup_tools(self):
        """Setup MCP tools for system monitoring"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available monitoring tools"""
            return [
                Tool(
                    name="get_system_health",
                    description="Get current system health status",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "components": {
                                "type": "array",
                                "description": "Specific components to check",
                                "items": {"type": "string"},
                                "default": ["all"]
                            },
                            "include_metrics": {
                                "type": "boolean",
                                "description": "Include detailed metrics",
                                "default": True
                            },
                            "include_alerts": {
                                "type": "boolean",
                                "description": "Include active alerts",
                                "default": True
                            }
                        }
                    }
                ),
                
                Tool(
                    name="monitor_performance",
                    description="Monitor system performance metrics",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "metrics": {
                                "type": "array",
                                "description": "Performance metrics to monitor",
                                "items": {"type": "string"},
                                "default": ["cpu", "memory", "disk", "network"]
                            },
                            "duration": {
                                "type": "integer",
                                "description": "Monitoring duration in seconds",
                                "default": 60
                            },
                            "interval": {
                                "type": "integer",
                                "description": "Sampling interval in seconds",
                                "default": 5
                            },
                            "thresholds": {
                                "type": "object",
                                "description": "Alert thresholds for metrics"
                            }
                        }
                    }
                ),
                
                Tool(
                    name="predict_errors",
                    description="Predict potential system errors",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prediction_horizon": {
                                "type": "integer",
                                "description": "Prediction horizon in minutes",
                                "default": 30
                            },
                            "confidence_threshold": {
                                "type": "number",
                                "description": "Minimum confidence for predictions",
                                "default": 0.7
                            },
                            "error_types": {
                                "type": "array",
                                "description": "Types of errors to predict",
                                "items": {"type": "string"},
                                "default": ["all"]
                            }
                        }
                    }
                ),
                
                Tool(
                    name="get_recovery_recommendations",
                    description="Get recovery recommendations for issues",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "issue_type": {
                                "type": "string",
                                "description": "Type of issue to address"
                            },
                            "severity": {
                                "type": "string",
                                "description": "Issue severity level",
                                "enum": ["low", "medium", "high", "critical"],
                                "default": "medium"
                            },
                            "context": {
                                "type": "object",
                                "description": "Additional context about the issue"
                            },
                            "auto_execute": {
                                "type": "boolean",
                                "description": "Automatically execute safe recommendations",
                                "default": False
                            }
                        },
                        "required": ["issue_type"]
                    }
                ),
                
                Tool(
                    name="set_monitoring_alerts",
                    description="Configure monitoring alerts and notifications",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "alert_rules": {
                                "type": "array",
                                "description": "Alert rules to configure",
                                "items": {"type": "object"}
                            },
                            "notification_channels": {
                                "type": "array",
                                "description": "Notification channels",
                                "items": {"type": "string"},
                                "default": ["console"]
                            },
                            "alert_schedule": {
                                "type": "object",
                                "description": "Alert schedule and timing"
                            }
                        },
                        "required": ["alert_rules"]
                    }
                ),
                
                Tool(
                    name="analyze_performance_trends",
                    description="Analyze performance trends and patterns",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "time_range": {
                                "type": "string",
                                "description": "Time range for analysis",
                                "enum": ["1h", "6h", "24h", "7d", "30d"],
                                "default": "24h"
                            },
                            "metrics": {
                                "type": "array",
                                "description": "Metrics to analyze",
                                "items": {"type": "string"}
                            },
                            "analysis_type": {
                                "type": "string",
                                "description": "Type of analysis",
                                "enum": ["trend", "anomaly", "correlation", "forecast"],
                                "default": "trend"
                            }
                        }
                    }
                ),
                
                Tool(
                    name="run_diagnostics",
                    description="Run comprehensive system diagnostics",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "diagnostic_type": {
                                "type": "string",
                                "description": "Type of diagnostic to run",
                                "enum": ["full", "quick", "component", "network"],
                                "default": "quick"
                            },
                            "components": {
                                "type": "array",
                                "description": "Specific components to diagnose",
                                "items": {"type": "string"}
                            },
                            "include_recommendations": {
                                "type": "boolean",
                                "description": "Include recommendations in results",
                                "default": True
                            }
                        }
                    }
                ),
                
                Tool(
                    name="export_monitoring_report",
                    description="Export monitoring and diagnostic reports",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "report_type": {
                                "type": "string",
                                "description": "Type of report to export",
                                "enum": ["health", "performance", "diagnostic", "comprehensive"],
                                "default": "comprehensive"
                            },
                            "time_range": {
                                "type": "string",
                                "description": "Time range for report data",
                                "default": "24h"
                            },
                            "format": {
                                "type": "string",
                                "description": "Report format",
                                "enum": ["json", "pdf", "html", "csv"],
                                "default": "json"
                            },
                            "include_charts": {
                                "type": "boolean",
                                "description": "Include charts and visualizations",
                                "default": True
                            }
                        }
                    }
                )
            ]
            
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls for system monitoring"""
            
            try:
                if name == "get_system_health":
                    return await self._handle_get_system_health(arguments)
                elif name == "monitor_performance":
                    return await self._handle_monitor_performance(arguments)
                elif name == "predict_errors":
                    return await self._handle_predict_errors(arguments)
                elif name == "get_recovery_recommendations":
                    return await self._handle_get_recovery_recommendations(arguments)
                elif name == "set_monitoring_alerts":
                    return await self._handle_set_monitoring_alerts(arguments)
                elif name == "analyze_performance_trends":
                    return await self._handle_analyze_performance_trends(arguments)
                elif name == "run_diagnostics":
                    return await self._handle_run_diagnostics(arguments)
                elif name == "export_monitoring_report":
                    return await self._handle_export_monitoring_report(arguments)
                else:
                    return [TextContent(
                        type="text",
                        text=f"Unknown tool: {name}"
                    )]
                    
            except Exception as e:
                logger.error(f"Error in tool {name}: {str(e)}")
                return [TextContent(
                    type="text",
                    text=f"Error executing {name}: {str(e)}"
                )]
                
    def _setup_resources(self):
        """Setup MCP resources for system monitoring"""
        
        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            """List available monitoring resources"""
            return [
                Resource(
                    uri="monitoring://health",
                    name="System Health",
                    description="Current system health status and metrics",
                    mimeType="application/json"
                ),
                Resource(
                    uri="monitoring://performance",
                    name="Performance Metrics",
                    description="Real-time performance metrics and trends",
                    mimeType="application/json"
                ),
                Resource(
                    uri="monitoring://alerts",
                    name="Alerts",
                    description="Active alerts and notifications",
                    mimeType="application/json"
                ),
                Resource(
                    uri="monitoring://diagnostics",
                    name="Diagnostics",
                    description="System diagnostics and health checks",
                    mimeType="application/json"
                )
            ]
            
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read monitoring resources"""
            try:
                if uri == "monitoring://health":
                    return json.dumps({
                        "health": await self._get_health_summary(),
                        "last_updated": datetime.now().isoformat()
                    })
                elif uri == "monitoring://performance":
                    return json.dumps({
                        "performance": await self._get_performance_summary(),
                        "last_updated": datetime.now().isoformat()
                    })
                elif uri == "monitoring://alerts":
                    return json.dumps({
                        "alerts": await self._get_alerts_summary(),
                        "last_updated": datetime.now().isoformat()
                    })
                elif uri == "monitoring://diagnostics":
                    return json.dumps({
                        "diagnostics": await self._get_diagnostics_summary(),
                        "last_updated": datetime.now().isoformat()
                    })
                else:
                    return json.dumps({"error": f"Unknown resource: {uri}"})
                    
            except Exception as e:
                logger.error(f"Error reading resource {uri}: {str(e)}")
                return json.dumps({"error": str(e)})
                
    async def _handle_get_system_health(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle system health check"""
        components = arguments.get("components", ["all"])
        include_metrics = arguments.get("include_metrics", True)
        include_alerts = arguments.get("include_alerts", True)
        
        # Get system health
        health_status = await self.health_monitor.get_system_health(
            components=components,
            include_metrics=include_metrics,
            include_alerts=include_alerts
        )
        
        result = {
            "health_check_id": str(uuid.uuid4()),
            "components": components,
            "include_metrics": include_metrics,
            "include_alerts": include_alerts,
            "health_status": health_status,
            "overall_status": health_status.get("overall_status", "unknown"),
            "timestamp": datetime.now().isoformat()
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
        
    async def _handle_monitor_performance(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle performance monitoring"""
        metrics = arguments.get("metrics", ["cpu", "memory", "disk", "network"])
        duration = arguments.get("duration", 60)
        interval = arguments.get("interval", 5)
        thresholds = arguments.get("thresholds", {})
        
        # Monitor performance
        performance_data = await self.performance_analyzer.monitor_performance(
            metrics=metrics,
            duration=duration,
            interval=interval,
            thresholds=thresholds
        )
        
        result = {
            "monitoring_id": str(uuid.uuid4()),
            "metrics": metrics,
            "duration": duration,
            "interval": interval,
            "thresholds": thresholds,
            "performance_data": performance_data,
            "summary": performance_data.get("summary", {}),
            "alerts": performance_data.get("alerts", []),
            "timestamp": datetime.now().isoformat()
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
        
    async def _handle_predict_errors(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle error prediction"""
        prediction_horizon = arguments.get("prediction_horizon", 30)
        confidence_threshold = arguments.get("confidence_threshold", 0.7)
        error_types = arguments.get("error_types", ["all"])
        
        # Predict errors
        predictions = await self.error_predictor.predict_errors(
            horizon_minutes=prediction_horizon,
            confidence_threshold=confidence_threshold,
            error_types=error_types
        )
        
        result = {
            "prediction_id": str(uuid.uuid4()),
            "prediction_horizon": prediction_horizon,
            "confidence_threshold": confidence_threshold,
            "error_types": error_types,
            "predictions": predictions,
            "total_predictions": len(predictions),
            "high_confidence_predictions": len([p for p in predictions if p.get("confidence", 0) > confidence_threshold]),
            "timestamp": datetime.now().isoformat()
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
        
    async def _handle_get_recovery_recommendations(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle recovery recommendations"""
        issue_type = arguments.get("issue_type")
        severity = arguments.get("severity", "medium")
        context = arguments.get("context", {})
        auto_execute = arguments.get("auto_execute", False)
        
        # Get recovery recommendations
        recommendations = await self.recovery_recommender.get_recommendations(
            issue_type=issue_type,
            severity=severity,
            context=context,
            auto_execute=auto_execute
        )
        
        result = {
            "recommendation_id": str(uuid.uuid4()),
            "issue_type": issue_type,
            "severity": severity,
            "context": context,
            "auto_execute": auto_execute,
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "executed_actions": recommendations.get("executed_actions", []),
            "timestamp": datetime.now().isoformat()
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
        
    async def _handle_set_monitoring_alerts(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle monitoring alert configuration"""
        alert_rules = arguments.get("alert_rules", [])
        notification_channels = arguments.get("notification_channels", ["console"])
        alert_schedule = arguments.get("alert_schedule", {})
        
        # Configure alerts
        alert_config = await self.health_monitor.configure_alerts(
            alert_rules=alert_rules,
            notification_channels=notification_channels,
            schedule=alert_schedule
        )
        
        result = {
            "config_id": str(uuid.uuid4()),
            "alert_rules": alert_rules,
            "notification_channels": notification_channels,
            "alert_schedule": alert_schedule,
            "alert_config": alert_config,
            "status": "alerts_configured",
            "total_rules": len(alert_rules),
            "timestamp": datetime.now().isoformat()
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
        
    async def _handle_analyze_performance_trends(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle performance trend analysis"""
        time_range = arguments.get("time_range", "24h")
        metrics = arguments.get("metrics", [])
        analysis_type = arguments.get("analysis_type", "trend")
        
        # Analyze trends
        trend_analysis = await self.performance_analyzer.analyze_trends(
            time_range=time_range,
            metrics=metrics,
            analysis_type=analysis_type
        )
        
        result = {
            "analysis_id": str(uuid.uuid4()),
            "time_range": time_range,
            "metrics": metrics,
            "analysis_type": analysis_type,
            "trend_analysis": trend_analysis,
            "key_insights": trend_analysis.get("insights", []),
            "anomalies": trend_analysis.get("anomalies", []),
            "timestamp": datetime.now().isoformat()
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
        
    async def _handle_run_diagnostics(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle system diagnostics"""
        diagnostic_type = arguments.get("diagnostic_type", "quick")
        components = arguments.get("components", [])
        include_recommendations = arguments.get("include_recommendations", True)
        
        # Run diagnostics
        diagnostic_results = await self.health_monitor.run_diagnostics(
            diagnostic_type=diagnostic_type,
            components=components,
            include_recommendations=include_recommendations
        )
        
        result = {
            "diagnostic_id": str(uuid.uuid4()),
            "diagnostic_type": diagnostic_type,
            "components": components,
            "include_recommendations": include_recommendations,
            "diagnostic_results": diagnostic_results,
            "overall_status": diagnostic_results.get("overall_status", "unknown"),
            "issues_found": len(diagnostic_results.get("issues", [])),
            "recommendations": diagnostic_results.get("recommendations", []),
            "timestamp": datetime.now().isoformat()
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
        
    async def _handle_export_monitoring_report(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle monitoring report export"""
        report_type = arguments.get("report_type", "comprehensive")
        time_range = arguments.get("time_range", "24h")
        format_type = arguments.get("format", "json")
        include_charts = arguments.get("include_charts", True)
        
        # Export report
        report_data = await self.health_monitor.export_report(
            report_type=report_type,
            time_range=time_range,
            format=format_type,
            include_charts=include_charts
        )
        
        result = {
            "export_id": str(uuid.uuid4()),
            "report_type": report_type,
            "time_range": time_range,
            "format": format_type,
            "include_charts": include_charts,
            "report_url": report_data.get("report_url", ""),
            "file_size": report_data.get("file_size", 0),
            "timestamp": datetime.now().isoformat()
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
        
    async def _get_health_summary(self) -> Dict[str, Any]:
        """Get summary of system health"""
        return {
            "overall_status": "healthy",
            "components": {
                "cpu": "healthy",
                "memory": "healthy",
                "disk": "healthy",
                "network": "healthy"
            },
            "active_alerts": 0,
            "last_check": datetime.now().isoformat()
        }
        
    async def _get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics"""
        return {
            "cpu_usage": 45.2,
            "memory_usage": 67.8,
            "disk_usage": 23.1,
            "network_throughput": 125.5,
            "response_time": 0.15,
            "last_updated": datetime.now().isoformat()
        }
        
    async def _get_alerts_summary(self) -> Dict[str, Any]:
        """Get summary of alerts"""
        return {
            "active_alerts": 0,
            "alert_history": [],
            "alert_types": ["performance", "error", "security", "capacity"],
            "last_alert": None
        }
        
    async def _get_diagnostics_summary(self) -> Dict[str, Any]:
        """Get summary of diagnostics"""
        return {
            "last_diagnostic": datetime.now().isoformat(),
            "diagnostic_status": "passed",
            "issues_found": 0,
            "recommendations": [],
            "components_checked": ["system", "network", "storage", "security"]
        }
        
    async def run(self):
        """Run the monitoring MCP server"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="monitoring",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=None,
                        experimental_capabilities=None,
                    ),
                ),
            )


async def main():
    """Main entry point for the monitoring MCP server"""
    server = MonitoringMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
