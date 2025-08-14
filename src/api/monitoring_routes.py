"""
Monitoring and Observability API Routes

This module provides API endpoints for the comprehensive monitoring system including:
- Application performance monitoring
- Infrastructure monitoring
- Business metrics monitoring
- Unified dashboard endpoints
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import statistics

from src.core.monitoring.application_monitor import application_monitor
from src.core.monitoring.infrastructure_monitor import infrastructure_monitor
from src.core.monitoring.business_metrics import business_metrics_monitor
from src.core.monitoring.alert_system import AlertSystem
from src.core.monitoring.decision_monitor import DecisionMonitor

router = APIRouter(prefix="/monitoring", tags=["Monitoring & Observability"])

# Initialize monitoring systems
alert_system = AlertSystem()
decision_monitor = DecisionMonitor()


@router.get("/status")
async def get_monitoring_status() -> Dict[str, Any]:
    """Get overall monitoring system status."""
    try:
        status = {
            "timestamp": datetime.now(),
            "application_monitoring": {
                "active": application_monitor.monitoring_active,
                "metrics_count": len(application_monitor.performance_metrics),
                "errors_count": len(application_monitor.error_records),
                "alerts_count": len(application_monitor.active_alerts)
            },
            "infrastructure_monitoring": {
                "active": infrastructure_monitor.monitoring_active,
                "servers_monitored": len(infrastructure_monitor.servers),
                "databases_monitored": len(infrastructure_monitor.databases),
                "alerts_count": len(infrastructure_monitor.infrastructure_alerts)
            },
            "business_metrics": {
                "active": business_metrics_monitor.monitoring_active,
                "decisions_count": len(business_metrics_monitor.decision_accuracy_metrics),
                "engagement_count": len(business_metrics_monitor.user_engagement_metrics),
                "alerts_count": len(business_metrics_monitor.business_alerts)
            },
            "alert_system": {
                "active_alerts": len(alert_system.active_alerts),
                "total_alerts": len(alert_system.alert_history)
            },
            "decision_monitor": {
                "active": decision_monitor.monitoring_active,
                "decisions_tracked": len(decision_monitor.decision_history)
            }
        }
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting monitoring status: {str(e)}")


@router.post("/start")
async def start_monitoring(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Start all monitoring systems."""
    try:
        # Start application monitoring
        application_monitor.start_monitoring()
        
        # Start infrastructure monitoring
        infrastructure_monitor.start_monitoring()
        
        # Start business metrics monitoring
        business_metrics_monitor.start_monitoring()
        
        # Start decision monitoring
        decision_monitor.start_monitoring()
        
        return {
            "status": "success",
            "message": "All monitoring systems started",
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting monitoring: {str(e)}")


@router.post("/stop")
async def stop_monitoring() -> Dict[str, Any]:
    """Stop all monitoring systems."""
    try:
        # Stop application monitoring
        application_monitor.stop_monitoring()
        
        # Stop infrastructure monitoring
        infrastructure_monitor.stop_monitoring()
        
        # Stop business metrics monitoring
        business_metrics_monitor.stop_monitoring()
        
        # Stop decision monitoring
        decision_monitor.stop_monitoring()
        
        return {
            "status": "success",
            "message": "All monitoring systems stopped",
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping monitoring: {str(e)}")


# Application Monitoring Endpoints
@router.get("/application/summary")
async def get_application_summary() -> Dict[str, Any]:
    """Get application performance summary."""
    try:
        summary = await application_monitor.get_performance_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting application summary: {str(e)}")


@router.get("/application/errors")
async def get_application_errors() -> Dict[str, Any]:
    """Get application error analysis."""
    try:
        errors = await application_monitor.get_error_analysis()
        return errors
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting error analysis: {str(e)}")


@router.get("/application/analytics")
async def get_application_analytics() -> Dict[str, Any]:
    """Get user analytics summary."""
    try:
        analytics = await application_monitor.get_user_analytics()
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting user analytics: {str(e)}")


@router.post("/application/record-error")
async def record_application_error(
    error_type: str,
    error_message: str,
    stack_trace: str = "",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    severity: str = "error"
) -> Dict[str, Any]:
    """Record an application error."""
    try:
        application_monitor.record_error(
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            user_id=user_id,
            session_id=session_id,
            severity=severity
        )
        return {"status": "success", "message": "Error recorded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recording error: {str(e)}")


@router.post("/application/record-action")
async def record_user_action(
    user_id: str,
    action: str,
    session_id: Optional[str] = None,
    page_url: Optional[str] = None,
    user_agent: Optional[str] = None,
    ip_address: Optional[str] = None
) -> Dict[str, Any]:
    """Record a user action."""
    try:
        application_monitor.record_user_action(
            user_id=user_id,
            action=action,
            session_id=session_id,
            page_url=page_url,
            user_agent=user_agent,
            ip_address=ip_address
        )
        return {"status": "success", "message": "User action recorded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recording user action: {str(e)}")


# Infrastructure Monitoring Endpoints
@router.get("/infrastructure/summary")
async def get_infrastructure_summary() -> Dict[str, Any]:
    """Get infrastructure monitoring summary."""
    try:
        summary = await infrastructure_monitor.get_infrastructure_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting infrastructure summary: {str(e)}")


@router.get("/infrastructure/servers/{server_id}")
async def get_server_status(server_id: str) -> Dict[str, Any]:
    """Get detailed status for a specific server."""
    try:
        status = await infrastructure_monitor.get_server_status(server_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting server status: {str(e)}")


@router.get("/infrastructure/databases/{database_id}")
async def get_database_status(database_id: str) -> Dict[str, Any]:
    """Get detailed status for a specific database."""
    try:
        status = await infrastructure_monitor.get_database_status(database_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting database status: {str(e)}")


@router.get("/infrastructure/network")
async def get_network_status() -> Dict[str, Any]:
    """Get network connectivity status."""
    try:
        status = await infrastructure_monitor.get_network_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting network status: {str(e)}")


# Business Metrics Endpoints
@router.get("/business/summary")
async def get_business_summary() -> Dict[str, Any]:
    """Get business metrics summary."""
    try:
        summary = await business_metrics_monitor.get_business_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting business summary: {str(e)}")


@router.get("/business/decisions")
async def get_decision_accuracy_report(hours: int = 24) -> Dict[str, Any]:
    """Get decision accuracy report."""
    try:
        report = await business_metrics_monitor.get_decision_accuracy_report(hours)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting decision report: {str(e)}")


@router.get("/business/engagement")
async def get_user_engagement_report(hours: int = 24) -> Dict[str, Any]:
    """Get user engagement report."""
    try:
        report = await business_metrics_monitor.get_user_engagement_report(hours)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting engagement report: {str(e)}")


@router.post("/business/record-decision")
async def record_decision_accuracy(
    decision_id: str,
    decision_type: str,
    predicted_outcome: Dict[str, Any],
    actual_outcome: Dict[str, Any],
    accuracy_score: float,
    confidence_score: float,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Record a decision accuracy metric."""
    try:
        business_metrics_monitor.record_decision_accuracy(
            decision_id=decision_id,
            decision_type=decision_type,
            predicted_outcome=predicted_outcome,
            actual_outcome=actual_outcome,
            accuracy_score=accuracy_score,
            confidence_score=confidence_score,
            user_id=user_id
        )
        return {"status": "success", "message": "Decision accuracy recorded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recording decision: {str(e)}")


@router.post("/business/record-engagement")
async def record_user_engagement(
    user_id: str,
    session_id: str,
    action_type: str,
    duration: float,
    page_url: Optional[str] = None,
    feature_used: Optional[str] = None,
    success: bool = True
) -> Dict[str, Any]:
    """Record a user engagement metric."""
    try:
        business_metrics_monitor.record_user_engagement(
            user_id=user_id,
            session_id=session_id,
            action_type=action_type,
            duration=duration,
            page_url=page_url,
            feature_used=feature_used,
            success=success
        )
        return {"status": "success", "message": "User engagement recorded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recording engagement: {str(e)}")


@router.post("/business/record-feature-usage")
async def record_feature_usage(
    feature_name: str,
    user_id: str,
    usage_count: int = 1,
    session_id: Optional[str] = None,
    success_rate: float = 1.0,
    average_duration: Optional[float] = None
) -> Dict[str, Any]:
    """Record a feature usage metric."""
    try:
        business_metrics_monitor.record_feature_usage(
            feature_name=feature_name,
            user_id=user_id,
            usage_count=usage_count,
            session_id=session_id,
            success_rate=success_rate,
            average_duration=average_duration
        )
        return {"status": "success", "message": "Feature usage recorded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recording feature usage: {str(e)}")


# Unified Dashboard Endpoints
@router.get("/dashboard/overview")
async def get_dashboard_overview() -> Dict[str, Any]:
    """Get comprehensive dashboard overview."""
    try:
        # Get all summaries concurrently
        app_summary, infra_summary, business_summary = await asyncio.gather(
            application_monitor.get_performance_summary(),
            infrastructure_monitor.get_infrastructure_summary(),
            business_metrics_monitor.get_business_summary(),
            return_exceptions=True
        )
        
        # Handle any exceptions
        if isinstance(app_summary, Exception):
            app_summary = {"error": str(app_summary)}
        if isinstance(infra_summary, Exception):
            infra_summary = {"error": str(infra_summary)}
        if isinstance(business_summary, Exception):
            business_summary = {"error": str(business_summary)}
        
        overview = {
            "timestamp": datetime.now(),
            "application": app_summary,
            "infrastructure": infra_summary,
            "business": business_summary,
            "system_health": {
                "overall_status": "healthy",
                "critical_alerts": 0,
                "warnings": 0
            }
        }
        
        # Calculate overall system health
        total_alerts = 0
        critical_alerts = 0
        warnings = 0
        
        # Count alerts from all systems
        if "alerts" in app_summary:
            total_alerts += app_summary["alerts"].get("active", 0)
        
        if "alerts" in infra_summary:
            recent_alerts = infra_summary["alerts"].get("recent", 0)
            total_alerts += recent_alerts
            if "by_severity" in infra_summary["alerts"]:
                critical_alerts += infra_summary["alerts"]["by_severity"].get("critical", 0)
                warnings += infra_summary["alerts"]["by_severity"].get("high", 0)
        
        if "alerts" in business_summary:
            recent_alerts = business_summary["alerts"].get("recent", 0)
            total_alerts += recent_alerts
            if "by_severity" in business_summary["alerts"]:
                critical_alerts += business_summary["alerts"]["by_severity"].get("critical", 0)
                warnings += business_summary["alerts"]["by_severity"].get("high", 0)
        
        overview["system_health"].update({
            "total_alerts": total_alerts,
            "critical_alerts": critical_alerts,
            "warnings": warnings,
            "overall_status": "critical" if critical_alerts > 0 else "warning" if warnings > 0 else "healthy"
        })
        
        return overview
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting dashboard overview: {str(e)}")


@router.get("/dashboard/metrics")
async def get_dashboard_metrics() -> Dict[str, Any]:
    """Get key metrics for dashboard visualization."""
    try:
        # Get recent metrics from all systems
        recent_cutoff = datetime.now() - timedelta(hours=1)
        
        # Application metrics
        app_metrics = [
            m for m in application_monitor.performance_metrics
            if m.timestamp > recent_cutoff
        ]
        
        # Infrastructure metrics
        infra_metrics = [
            m for m in infrastructure_monitor.server_metrics
            if m.timestamp > recent_cutoff
        ]
        
        # Business metrics
        business_metrics = [
            m for m in business_metrics_monitor.decision_accuracy_metrics
            if m.timestamp > recent_cutoff
        ]
        
        # Calculate key performance indicators
        kpis = {
            "timestamp": datetime.now(),
            "application": {
                "cpu_usage": statistics.mean([m.value for m in app_metrics if m.metric_name == "cpu_usage"]) if app_metrics else 0,
                "memory_usage": statistics.mean([m.value for m in app_metrics if m.metric_name == "memory_usage"]) if app_metrics else 0,
                "error_rate": len([e for e in application_monitor.error_records if e.timestamp > recent_cutoff]) / 60  # errors per minute
            },
            "infrastructure": {
                "server_health": len([m for m in infra_metrics if m.value < 80]) / len(infra_metrics) if infra_metrics else 1.0,
                "network_latency": statistics.mean([m.value for m in infrastructure_monitor.network_metrics if m.metric_name == "ping_latency" and m.timestamp > recent_cutoff]) if infrastructure_monitor.network_metrics else 0
            },
            "business": {
                "decision_accuracy": statistics.mean([m.accuracy_score for m in business_metrics]) if business_metrics else 0,
                "user_engagement": len([e for e in business_metrics_monitor.user_engagement_metrics if e.timestamp > recent_cutoff])
            }
        }
        
        return kpis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting dashboard metrics: {str(e)}")


# Alert Management Endpoints
@router.get("/alerts")
async def get_all_alerts() -> Dict[str, Any]:
    """Get all active alerts from all monitoring systems."""
    try:
        alerts = {
            "timestamp": datetime.now(),
            "application_alerts": list(application_monitor.active_alerts.values()),
            "infrastructure_alerts": [
                {
                    "alert_id": a.alert_id,
                    "alert_type": a.alert_type,
                    "severity": a.severity,
                    "message": a.message,
                    "timestamp": a.timestamp,
                    "source_id": a.source_id
                }
                for a in infrastructure_monitor.infrastructure_alerts
                if a.timestamp > datetime.now() - timedelta(hours=24)
            ],
            "business_alerts": [
                {
                    "alert_id": a.alert_id,
                    "alert_type": a.alert_type,
                    "severity": a.severity,
                    "message": a.message,
                    "timestamp": a.timestamp
                }
                for a in business_metrics_monitor.business_alerts
                if a.timestamp > datetime.now() - timedelta(hours=24)
            ]
        }
        
        # Calculate totals
        alerts["total_alerts"] = (
            len(alerts["application_alerts"]) +
            len(alerts["infrastructure_alerts"]) +
            len(alerts["business_alerts"])
        )
        
        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting alerts: {str(e)}")


@router.get("/alerts/{alert_type}")
async def get_alerts_by_type(alert_type: str) -> Dict[str, Any]:
    """Get alerts by type (application, infrastructure, business)."""
    try:
        if alert_type == "application":
            alerts = list(application_monitor.active_alerts.values())
        elif alert_type == "infrastructure":
            alerts = [
                {
                    "alert_id": a.alert_id,
                    "alert_type": a.alert_type,
                    "severity": a.severity,
                    "message": a.message,
                    "timestamp": a.timestamp,
                    "source_id": a.source_id
                }
                for a in infrastructure_monitor.infrastructure_alerts
                if a.timestamp > datetime.now() - timedelta(hours=24)
            ]
        elif alert_type == "business":
            alerts = [
                {
                    "alert_id": a.alert_id,
                    "alert_type": a.alert_type,
                    "severity": a.severity,
                    "message": a.message,
                    "timestamp": a.timestamp
                }
                for a in business_metrics_monitor.business_alerts
                if a.timestamp > datetime.now() - timedelta(hours=24)
            ]
        else:
            raise HTTPException(status_code=400, detail=f"Invalid alert type: {alert_type}")
        
        return {
            "alert_type": alert_type,
            "timestamp": datetime.now(),
            "alerts": alerts,
            "count": len(alerts)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting alerts: {str(e)}")


@router.delete("/alerts/{alert_id}")
async def clear_alert(alert_id: str) -> Dict[str, Any]:
    """Clear a specific alert."""
    try:
        # Try to clear from application alerts
        if alert_id in application_monitor.active_alerts:
            del application_monitor.active_alerts[alert_id]
            return {"status": "success", "message": f"Alert {alert_id} cleared"}
        
        # Try to clear from infrastructure alerts
        for alert in infrastructure_monitor.infrastructure_alerts:
            if alert.alert_id == alert_id:
                infrastructure_monitor.infrastructure_alerts.remove(alert)
                return {"status": "success", "message": f"Alert {alert_id} cleared"}
        
        # Try to clear from business alerts
        for alert in business_metrics_monitor.business_alerts:
            if alert.alert_id == alert_id:
                business_metrics_monitor.business_alerts.remove(alert)
                return {"status": "success", "message": f"Alert {alert_id} cleared"}
        
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing alert: {str(e)}")


# Configuration Endpoints
@router.get("/config")
async def get_monitoring_config() -> Dict[str, Any]:
    """Get monitoring system configuration."""
    try:
        config = {
            "timestamp": datetime.now(),
            "application_monitoring": {
                "monitoring_interval": application_monitor.monitoring_interval,
                "alert_rules_count": len(application_monitor.alert_rules)
            },
            "infrastructure_monitoring": {
                "monitoring_interval": infrastructure_monitor.monitoring_interval,
                "servers": list(infrastructure_monitor.servers.keys()),
                "databases": list(infrastructure_monitor.databases.keys()),
                "network_targets": infrastructure_monitor.network_targets,
                "alert_thresholds": infrastructure_monitor.alert_thresholds
            },
            "business_metrics": {
                "monitoring_interval": business_metrics_monitor.monitoring_interval,
                "features": list(business_metrics_monitor.features.keys()),
                "business_thresholds": business_metrics_monitor.business_thresholds
            }
        }
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting config: {str(e)}")


@router.post("/config/thresholds")
async def update_alert_thresholds(
    system: str,
    metric_name: str,
    threshold: float
) -> Dict[str, Any]:
    """Update alert thresholds for a specific system."""
    try:
        if system == "infrastructure":
            infrastructure_monitor.alert_thresholds[metric_name] = threshold
        elif system == "business":
            business_metrics_monitor.business_thresholds[metric_name] = threshold
        else:
            raise HTTPException(status_code=400, detail=f"Invalid system: {system}")
        
        return {
            "status": "success",
            "message": f"Threshold updated for {system}.{metric_name}",
            "new_threshold": threshold
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating threshold: {str(e)}")
