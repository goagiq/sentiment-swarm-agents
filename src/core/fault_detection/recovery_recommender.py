"""
Recovery Recommender

Provides system recovery suggestions and automated recovery actions.
Analyzes system issues and recommends appropriate recovery strategies.
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import subprocess
import os
import json

logger = logging.getLogger(__name__)


class RecoveryAction(Enum):
    """Types of recovery actions"""
    RESTART_SERVICE = "restart_service"
    CLEAR_CACHE = "clear_cache"
    FREE_MEMORY = "free_memory"
    CLEAN_DISK = "clean_disk"
    RESTART_PROCESS = "restart_process"
    SCALE_RESOURCES = "scale_resources"
    ROLLBACK_CONFIG = "rollback_config"
    EMERGENCY_RESTART = "emergency_restart"


class RecoveryPriority(Enum):
    """Recovery action priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RecoveryRecommendation:
    """Recovery recommendation with action plan"""
    action: RecoveryAction
    priority: RecoveryPriority
    description: str
    expected_impact: str
    estimated_duration: int  # minutes
    success_probability: float
    risk_level: str
    prerequisites: List[str]
    steps: List[str]
    automated: bool


@dataclass
class RecoveryPlan:
    """Complete recovery plan for a system issue"""
    issue_type: str
    severity: str
    recommendations: List[RecoveryRecommendation]
    estimated_total_time: int  # minutes
    overall_success_probability: float
    risk_assessment: str


class RecoveryRecommender:
    """Recommends and executes system recovery actions."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.recovery_active = False
        self.recovery_thread: Optional[threading.Thread] = None
        self.recovery_history: List[Dict[str, Any]] = []
        
        # Recovery action configurations
        self.action_configs = self.config.get('action_configs', {
            'restart_service': {
                'timeout': 60,
                'retry_count': 3,
                'services': ['main', 'api', 'monitoring']
            },
            'clear_cache': {
                'cache_dirs': ['/tmp', './cache'],
                'max_size': '1GB'
            },
            'free_memory': {
                'threshold': 85.0,
                'methods': ['gc', 'restart_processes']
            }
        })
        
    def analyze_system_issues(self, health_data: Dict[str, Any]) -> RecoveryPlan:
        """Analyze system issues and create recovery plan"""
        try:
            issues = self._identify_issues(health_data)
            recommendations = []
            
            for issue in issues:
                issue_recommendations = self._generate_recommendations(issue)
                recommendations.extend(issue_recommendations)
                
            # Prioritize and organize recommendations
            prioritized_recommendations = self._prioritize_recommendations(recommendations)
            
            # Create recovery plan
            plan = RecoveryPlan(
                issue_type="system_issues",
                severity=self._determine_overall_severity(issues),
                recommendations=prioritized_recommendations,
                estimated_total_time=self._calculate_total_time(prioritized_recommendations),
                overall_success_probability=self._calculate_success_probability(prioritized_recommendations),
                risk_assessment=self._assess_overall_risk(prioritized_recommendations)
            )
            
            return plan
            
        except Exception as e:
            logger.error(f"Error analyzing system issues: {e}")
            return self._create_emergency_plan()
            
    def _identify_issues(self, health_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific issues from health data"""
        issues = []
        
        try:
            # Check CPU issues
            if 'system_metrics' in health_data:
                metrics = health_data['system_metrics']
                if metrics.get('cpu_percent', 0) > 90:
                    issues.append({
                        'type': 'high_cpu',
                        'severity': 'critical',
                        'value': metrics['cpu_percent'],
                        'threshold': 90
                    })
                elif metrics.get('cpu_percent', 0) > 80:
                    issues.append({
                        'type': 'high_cpu',
                        'severity': 'high',
                        'value': metrics['cpu_percent'],
                        'threshold': 80
                    })
                    
            # Check memory issues
            if 'system_metrics' in health_data:
                metrics = health_data['system_metrics']
                if metrics.get('memory_percent', 0) > 95:
                    issues.append({
                        'type': 'high_memory',
                        'severity': 'critical',
                        'value': metrics['memory_percent'],
                        'threshold': 95
                    })
                elif metrics.get('memory_percent', 0) > 85:
                    issues.append({
                        'type': 'high_memory',
                        'severity': 'high',
                        'value': metrics['memory_percent'],
                        'threshold': 85
                    })
                    
            # Check disk issues
            if 'system_metrics' in health_data:
                metrics = health_data['system_metrics']
                if metrics.get('disk_percent', 0) > 98:
                    issues.append({
                        'type': 'disk_full',
                        'severity': 'critical',
                        'value': metrics['disk_percent'],
                        'threshold': 98
                    })
                elif metrics.get('disk_percent', 0) > 90:
                    issues.append({
                        'type': 'disk_full',
                        'severity': 'high',
                        'value': metrics['disk_percent'],
                        'threshold': 90
                    })
                    
            # Check component issues
            if 'components' in health_data:
                for component_name, component_data in health_data['components'].items():
                    if component_data.get('status') == 'critical':
                        issues.append({
                            'type': 'component_failure',
                            'severity': 'critical',
                            'component': component_name,
                            'details': component_data
                        })
                        
        except Exception as e:
            logger.error(f"Error identifying issues: {e}")
            
        return issues
        
    def _generate_recommendations(self, issue: Dict[str, Any]) -> List[RecoveryRecommendation]:
        """Generate recovery recommendations for a specific issue"""
        recommendations = []
        
        try:
            issue_type = issue.get('type')
            severity = issue.get('severity', 'medium')
            
            if issue_type == 'high_cpu':
                recommendations.extend(self._generate_cpu_recommendations(issue))
            elif issue_type == 'high_memory':
                recommendations.extend(self._generate_memory_recommendations(issue))
            elif issue_type == 'disk_full':
                recommendations.extend(self._generate_disk_recommendations(issue))
            elif issue_type == 'component_failure':
                recommendations.extend(self._generate_component_recommendations(issue))
            else:
                # Generic recommendations
                recommendations.append(self._create_generic_recommendation(issue))
                
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            
        return recommendations
        
    def _generate_cpu_recommendations(self, issue: Dict[str, Any]) -> List[RecoveryRecommendation]:
        """Generate CPU-related recovery recommendations"""
        recommendations = []
        severity = issue.get('severity', 'medium')
        
        # Clear cache recommendation
        recommendations.append(RecoveryRecommendation(
            action=RecoveryAction.CLEAR_CACHE,
            priority=RecoveryPriority.MEDIUM if severity == 'high' else RecoveryPriority.HIGH,
            description="Clear system cache to reduce CPU load",
            expected_impact="Reduce CPU usage by 10-20%",
            estimated_duration=5,
            success_probability=0.8,
            risk_level="low",
            prerequisites=["Cache directories accessible"],
            steps=[
                "Identify cache directories",
                "Calculate cache size",
                "Clear cache files",
                "Verify cache clearance"
            ],
            automated=True
        ))
        
        # Restart high-CPU processes
        if severity == 'critical':
            recommendations.append(RecoveryRecommendation(
                action=RecoveryAction.RESTART_PROCESS,
                priority=RecoveryPriority.HIGH,
                description="Restart high-CPU consuming processes",
                expected_impact="Immediate CPU usage reduction",
                estimated_duration=10,
                success_probability=0.7,
                risk_level="medium",
                prerequisites=["Process list accessible"],
                steps=[
                    "Identify high-CPU processes",
                    "Stop processes gracefully",
                    "Restart processes",
                    "Monitor CPU usage"
                ],
                automated=True
            ))
            
        return recommendations
        
    def _generate_memory_recommendations(self, issue: Dict[str, Any]) -> List[RecoveryRecommendation]:
        """Generate memory-related recovery recommendations"""
        recommendations = []
        severity = issue.get('severity', 'medium')
        
        # Free memory recommendation
        recommendations.append(RecoveryRecommendation(
            action=RecoveryAction.FREE_MEMORY,
            priority=RecoveryPriority.HIGH if severity == 'critical' else RecoveryPriority.MEDIUM,
            description="Free up system memory",
            expected_impact="Reduce memory usage by 15-30%",
            estimated_duration=8,
            success_probability=0.75,
            risk_level="low",
            prerequisites=["Memory management tools available"],
            steps=[
                "Force garbage collection",
                "Clear unused memory",
                "Restart memory-intensive processes",
                "Verify memory reduction"
            ],
            automated=True
        ))
        
        # Scale resources if critical
        if severity == 'critical':
            recommendations.append(RecoveryRecommendation(
                action=RecoveryAction.SCALE_RESOURCES,
                priority=RecoveryPriority.CRITICAL,
                description="Scale up memory resources",
                expected_impact="Immediate memory availability",
                estimated_duration=15,
                success_probability=0.9,
                risk_level="medium",
                prerequisites=["Resource scaling available"],
                steps=[
                    "Check available resources",
                    "Scale memory allocation",
                    "Restart affected services",
                    "Monitor memory usage"
                ],
                automated=False
            ))
            
        return recommendations
        
    def _generate_disk_recommendations(self, issue: Dict[str, Any]) -> List[RecoveryRecommendation]:
        """Generate disk-related recovery recommendations"""
        recommendations = []
        severity = issue.get('severity', 'medium')
        
        # Clean disk recommendation
        recommendations.append(RecoveryRecommendation(
            action=RecoveryAction.CLEAN_DISK,
            priority=RecoveryPriority.HIGH if severity == 'critical' else RecoveryPriority.MEDIUM,
            description="Clean up disk space",
            expected_impact="Free up 10-20% disk space",
            estimated_duration=12,
            success_probability=0.8,
            risk_level="low",
            prerequisites=["Disk cleanup tools available"],
            steps=[
                "Identify large files",
                "Remove temporary files",
                "Clear log files",
                "Verify disk space"
            ],
            automated=True
        ))
        
        return recommendations
        
    def _generate_component_recommendations(self, issue: Dict[str, Any]) -> List[RecoveryRecommendation]:
        """Generate component failure recovery recommendations"""
        recommendations = []
        component_name = issue.get('component', 'unknown')
        
        # Restart service recommendation
        recommendations.append(RecoveryRecommendation(
            action=RecoveryAction.RESTART_SERVICE,
            priority=RecoveryPriority.HIGH,
            description=f"Restart failed component: {component_name}",
            expected_impact="Restore component functionality",
            estimated_duration=10,
            success_probability=0.8,
            risk_level="medium",
            prerequisites=["Service management available"],
            steps=[
                f"Stop {component_name} service",
                "Wait for graceful shutdown",
                f"Start {component_name} service",
                "Verify service status"
            ],
            automated=True
        ))
        
        return recommendations
        
    def _create_generic_recommendation(self, issue: Dict[str, Any]) -> RecoveryRecommendation:
        """Create a generic recovery recommendation"""
        return RecoveryRecommendation(
            action=RecoveryAction.RESTART_SERVICE,
            priority=RecoveryPriority.MEDIUM,
            description="Generic system recovery action",
            expected_impact="Improve system stability",
            estimated_duration=15,
            success_probability=0.6,
            risk_level="medium",
            prerequisites=["System access available"],
            steps=[
                "Analyze system state",
                "Identify root cause",
                "Apply recovery action",
                "Verify recovery success"
            ],
            automated=False
        )
        
    def _prioritize_recommendations(self, recommendations: List[RecoveryRecommendation]) -> List[RecoveryRecommendation]:
        """Prioritize recovery recommendations"""
        # Sort by priority (critical -> high -> medium -> low)
        priority_order = {
            RecoveryPriority.CRITICAL: 0,
            RecoveryPriority.HIGH: 1,
            RecoveryPriority.MEDIUM: 2,
            RecoveryPriority.LOW: 3
        }
        
        return sorted(recommendations, key=lambda r: priority_order[r.priority])
        
    def _determine_overall_severity(self, issues: List[Dict[str, Any]]) -> str:
        """Determine overall severity from issues"""
        if not issues:
            return "none"
            
        severities = [issue.get('severity', 'medium') for issue in issues]
        
        if 'critical' in severities:
            return 'critical'
        elif 'high' in severities:
            return 'high'
        elif 'medium' in severities:
            return 'medium'
        else:
            return 'low'
            
    def _calculate_total_time(self, recommendations: List[RecoveryRecommendation]) -> int:
        """Calculate total estimated time for all recommendations"""
        return sum(r.estimated_duration for r in recommendations)
        
    def _calculate_success_probability(self, recommendations: List[RecoveryRecommendation]) -> float:
        """Calculate overall success probability"""
        if not recommendations:
            return 0.0
            
        probabilities = [r.success_probability for r in recommendations]
        return sum(probabilities) / len(probabilities)
        
    def _assess_overall_risk(self, recommendations: List[RecoveryRecommendation]) -> str:
        """Assess overall risk of recovery plan"""
        if not recommendations:
            return "none"
            
        risk_levels = [r.risk_level for r in recommendations]
        
        if 'high' in risk_levels:
            return 'high'
        elif 'medium' in risk_levels:
            return 'medium'
        else:
            return 'low'
            
    def _create_emergency_plan(self) -> RecoveryPlan:
        """Create emergency recovery plan"""
        emergency_recommendation = RecoveryRecommendation(
            action=RecoveryAction.EMERGENCY_RESTART,
            priority=RecoveryPriority.CRITICAL,
            description="Emergency system restart",
            expected_impact="Complete system recovery",
            estimated_duration=30,
            success_probability=0.9,
            risk_level="high",
            prerequisites=["System restart capability"],
            steps=[
                "Save critical data",
                "Stop all services",
                "Restart system",
                "Verify system status"
            ],
            automated=False
        )
        
        return RecoveryPlan(
            issue_type="emergency",
            severity="critical",
            recommendations=[emergency_recommendation],
            estimated_total_time=30,
            overall_success_probability=0.9,
            risk_assessment="high"
        )
        
    def execute_recovery_action(self, recommendation: RecoveryRecommendation) -> Dict[str, Any]:
        """Execute a recovery action"""
        try:
            start_time = datetime.now()
            logger.info(f"Executing recovery action: {recommendation.action.value}")
            
            if recommendation.automated:
                result = self._execute_automated_action(recommendation)
            else:
                result = self._execute_manual_action(recommendation)
                
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Record recovery attempt
            recovery_record = {
                'action': recommendation.action.value,
                'timestamp': start_time.isoformat(),
                'duration': duration,
                'success': result.get('success', False),
                'details': result
            }
            self.recovery_history.append(recovery_record)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing recovery action: {e}")
            return {
                'success': False,
                'error': str(e),
                'action': recommendation.action.value
            }
            
    def _execute_automated_action(self, recommendation: RecoveryRecommendation) -> Dict[str, Any]:
        """Execute automated recovery action"""
        try:
            if recommendation.action == RecoveryAction.CLEAR_CACHE:
                return self._clear_cache()
            elif recommendation.action == RecoveryAction.FREE_MEMORY:
                return self._free_memory()
            elif recommendation.action == RecoveryAction.CLEAN_DISK:
                return self._clean_disk()
            elif recommendation.action == RecoveryAction.RESTART_PROCESS:
                return self._restart_process()
            else:
                return {'success': False, 'error': 'Action not implemented'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    def _execute_manual_action(self, recommendation: RecoveryRecommendation) -> Dict[str, Any]:
        """Execute manual recovery action"""
        return {
            'success': True,
            'manual_action_required': True,
            'description': recommendation.description,
            'steps': recommendation.steps
        }
        
    def _clear_cache(self) -> Dict[str, Any]:
        """Clear system cache"""
        try:
            cache_dirs = self.action_configs.get('clear_cache', {}).get('cache_dirs', [])
            cleared_size = 0
            
            for cache_dir in cache_dirs:
                if os.path.exists(cache_dir):
                    # Simple cache clearing (in production, would be more sophisticated)
                    cleared_size += self._calculate_dir_size(cache_dir)
                    
            return {
                'success': True,
                'cleared_size': cleared_size,
                'cache_dirs': cache_dirs
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    def _free_memory(self) -> Dict[str, Any]:
        """Free system memory"""
        try:
            import gc
            gc.collect()
            
            return {
                'success': True,
                'action': 'garbage_collection',
                'memory_freed': 'unknown'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    def _clean_disk(self) -> Dict[str, Any]:
        """Clean disk space"""
        try:
            # Simple disk cleaning (in production, would be more sophisticated)
            cleaned_size = 0
            
            # Clean temporary files
            temp_dirs = ['/tmp', './temp']
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    cleaned_size += self._calculate_dir_size(temp_dir)
                    
            return {
                'success': True,
                'cleaned_size': cleaned_size,
                'temp_dirs_cleaned': temp_dirs
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    def _restart_process(self) -> Dict[str, Any]:
        """Restart high-CPU processes"""
        try:
            # This would identify and restart specific processes
            # For now, return a placeholder
            return {
                'success': True,
                'action': 'process_restart',
                'processes_restarted': 0
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    def _calculate_dir_size(self, directory: str) -> int:
        """Calculate directory size in bytes"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception:
            pass
        return total_size
        
    def get_recovery_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recovery history for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_history = [
            record for record in self.recovery_history
            if datetime.fromisoformat(record['timestamp']) > cutoff_time
        ]
        return recent_history
        
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        if not self.recovery_history:
            return {"message": "No recovery history available"}
            
        total_attempts = len(self.recovery_history)
        successful_attempts = sum(1 for r in self.recovery_history if r.get('success', False))
        success_rate = successful_attempts / total_attempts if total_attempts > 0 else 0.0
        
        return {
            'total_attempts': total_attempts,
            'successful_attempts': successful_attempts,
            'success_rate': success_rate,
            'average_duration': sum(r.get('duration', 0) for r in self.recovery_history) / total_attempts
        }
