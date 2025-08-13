"""
Implementation Planner Component

Provides step-by-step action planning and implementation guidance
for recommendations with timeline, resource allocation, and milestone tracking.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime, timedelta

from .recommendation_engine import Recommendation

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of implementation tasks."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Priority levels for implementation tasks."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskType(Enum):
    """Types of implementation tasks."""
    PLANNING = "planning"
    ANALYSIS = "analysis"
    DEVELOPMENT = "development"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    TRAINING = "training"
    DOCUMENTATION = "documentation"
    REVIEW = "review"


@dataclass
class ImplementationTask:
    """Represents a single implementation task."""
    id: str
    title: str
    description: str
    task_type: TaskType
    priority: TaskPriority
    status: TaskStatus = TaskStatus.NOT_STARTED
    estimated_duration_days: int = 1
    actual_duration_days: Optional[int] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    assigned_to: Optional[str] = None
    resources_required: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)
    notes: str = ""
    progress_percentage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization."""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'task_type': self.task_type.value,
            'priority': self.priority.value,
            'status': self.status.value,
            'estimated_duration_days': self.estimated_duration_days,
            'actual_duration_days': self.actual_duration_days,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'dependencies': self.dependencies,
            'assigned_to': self.assigned_to,
            'resources_required': self.resources_required,
            'deliverables': self.deliverables,
            'notes': self.notes,
            'progress_percentage': self.progress_percentage
        }


@dataclass
class ImplementationPhase:
    """Represents a phase of implementation."""
    name: str
    description: str
    tasks: List[ImplementationTask] = field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    status: TaskStatus = TaskStatus.NOT_STARTED
    progress_percentage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert phase to dictionary for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'tasks': [task.to_dict() for task in self.tasks],
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'status': self.status.value,
            'progress_percentage': self.progress_percentage
        }


@dataclass
class ImplementationPlan:
    """Complete implementation plan for a recommendation."""
    recommendation_id: str
    recommendation_title: str
    phases: List[ImplementationPhase] = field(default_factory=list)
    total_duration_days: int = 0
    total_estimated_cost: Optional[float] = None
    risk_factors: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary for serialization."""
        return {
            'recommendation_id': self.recommendation_id,
            'recommendation_title': self.recommendation_title,
            'phases': [phase.to_dict() for phase in self.phases],
            'total_duration_days': self.total_duration_days,
            'total_estimated_cost': self.total_estimated_cost,
            'risk_factors': self.risk_factors,
            'success_criteria': self.success_criteria,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class PlanningContext:
    """Context information for implementation planning."""
    available_resources: Dict[str, Any] = field(default_factory=dict)
    team_capacity: Dict[str, int] = field(default_factory=dict)
    budget_constraints: Optional[float] = None
    timeline_constraints: Optional[int] = None
    risk_tolerance: str = "medium"
    stakeholder_requirements: List[str] = field(default_factory=list)
    technical_constraints: List[str] = field(default_factory=list)


class ImplementationPlanner:
    """Intelligent implementation planning engine."""
    
    def __init__(self):
        self.task_templates = self._load_task_templates()
        self.phase_templates = self._load_phase_templates()
        logger.info("Initialized ImplementationPlanner")
    
    def _load_task_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load task templates for different recommendation types."""
        return {
            "technology_adoption": {
                "planning": {
                    "title": "Technology Assessment and Planning",
                    "description": "Evaluate technology requirements and create implementation plan",
                    "duration": 5,
                    "dependencies": [],
                    "deliverables": ["Technology assessment report", "Implementation plan"]
                },
                "development": {
                    "title": "Technology Implementation",
                    "description": "Implement the selected technology solution",
                    "duration": 15,
                    "dependencies": ["planning"],
                    "deliverables": ["Working technology solution"]
                },
                "testing": {
                    "title": "Testing and Validation",
                    "description": "Test the implemented solution and validate functionality",
                    "duration": 7,
                    "dependencies": ["development"],
                    "deliverables": ["Test results", "Validation report"]
                },
                "deployment": {
                    "title": "Production Deployment",
                    "description": "Deploy the solution to production environment",
                    "duration": 3,
                    "dependencies": ["testing"],
                    "deliverables": ["Production deployment"]
                }
            },
            "process_improvement": {
                "analysis": {
                    "title": "Current Process Analysis",
                    "description": "Analyze current processes and identify improvement opportunities",
                    "duration": 7,
                    "dependencies": [],
                    "deliverables": ["Process analysis report", "Improvement recommendations"]
                },
                "design": {
                    "title": "Process Redesign",
                    "description": "Design improved processes and workflows",
                    "duration": 10,
                    "dependencies": ["analysis"],
                    "deliverables": ["Process design document", "Workflow diagrams"]
                },
                "implementation": {
                    "title": "Process Implementation",
                    "description": "Implement the new processes and workflows",
                    "duration": 14,
                    "dependencies": ["design"],
                    "deliverables": ["Implemented processes", "Training materials"]
                },
                "monitoring": {
                    "title": "Process Monitoring and Optimization",
                    "description": "Monitor process performance and optimize as needed",
                    "duration": 5,
                    "dependencies": ["implementation"],
                    "deliverables": ["Performance metrics", "Optimization recommendations"]
                }
            },
            "risk_management": {
                "assessment": {
                    "title": "Risk Assessment",
                    "description": "Identify and assess potential risks",
                    "duration": 5,
                    "dependencies": [],
                    "deliverables": ["Risk assessment report", "Risk register"]
                },
                "mitigation_planning": {
                    "title": "Risk Mitigation Planning",
                    "description": "Develop risk mitigation strategies and plans",
                    "duration": 7,
                    "dependencies": ["assessment"],
                    "deliverables": ["Risk mitigation plan", "Contingency plans"]
                },
                "implementation": {
                    "title": "Risk Mitigation Implementation",
                    "description": "Implement risk mitigation measures",
                    "duration": 12,
                    "dependencies": ["mitigation_planning"],
                    "deliverables": ["Implemented risk controls", "Monitoring systems"]
                },
                "monitoring": {
                    "title": "Risk Monitoring and Review",
                    "description": "Monitor risk levels and review effectiveness",
                    "duration": 3,
                    "dependencies": ["implementation"],
                    "deliverables": ["Risk monitoring report", "Effectiveness review"]
                }
            }
        }
    
    def _load_phase_templates(self) -> Dict[str, List[str]]:
        """Load phase templates for different recommendation types."""
        return {
            "technology_adoption": ["Planning", "Development", "Testing", "Deployment"],
            "process_improvement": ["Analysis", "Design", "Implementation", "Monitoring"],
            "risk_management": ["Assessment", "Mitigation Planning", "Implementation", "Monitoring"],
            "market_strategy": ["Research", "Strategy Development", "Implementation", "Evaluation"],
            "performance_optimization": ["Baseline Assessment", "Optimization Design", "Implementation", "Performance Monitoring"]
        }
    
    async def create_implementation_plan(
        self,
        recommendation: Recommendation,
        context: PlanningContext
    ) -> ImplementationPlan:
        """Create a comprehensive implementation plan for a recommendation."""
        try:
            logger.info(f"Creating implementation plan for: {recommendation.title}")
            
            # Determine recommendation type and get appropriate templates
            recommendation_type = self._get_recommendation_type(recommendation)
            task_template = self.task_templates.get(recommendation_type, {})
            phase_names = self.phase_templates.get(recommendation_type, ["Planning", "Implementation", "Review"])
            
            # Create phases and tasks
            phases = []
            current_date = datetime.now()
            total_duration = 0
            
            for i, phase_name in enumerate(phase_names):
                phase = ImplementationPhase(
                    name=phase_name,
                    description=f"Phase {i+1}: {phase_name}",
                    start_date=current_date + timedelta(days=total_duration)
                )
                
                # Create tasks for this phase
                phase_tasks = self._create_phase_tasks(
                    phase_name, task_template, recommendation, context
                )
                phase.tasks = phase_tasks
                
                # Calculate phase duration and end date
                phase_duration = sum(task.estimated_duration_days for task in phase_tasks)
                phase.end_date = phase.start_date + timedelta(days=phase_duration)
                total_duration += phase_duration
                
                phases.append(phase)
            
            # Create implementation plan
            plan = ImplementationPlan(
                recommendation_id=recommendation.id,
                recommendation_title=recommendation.title,
                phases=phases,
                total_duration_days=total_duration,
                total_estimated_cost=self._estimate_total_cost(recommendation, phases),
                risk_factors=self._identify_risk_factors(recommendation, context),
                success_criteria=self._define_success_criteria(recommendation)
            )
            
            logger.info(f"Created implementation plan with {len(phases)} phases and {sum(len(p.tasks) for p in phases)} tasks")
            return plan
            
        except Exception as e:
            logger.error(f"Error creating implementation plan: {e}")
            return ImplementationPlan(
                recommendation_id=recommendation.id,
                recommendation_title=recommendation.title
            )
    
    def _get_recommendation_type(self, recommendation: Recommendation) -> str:
        """Determine the recommendation type for template selection."""
        category_mapping = {
            "technology_adoption": "technology_adoption",
            "process_improvement": "process_improvement",
            "risk_management": "risk_management",
            "market_strategy": "market_strategy",
            "performance_optimization": "performance_optimization"
        }
        
        return category_mapping.get(recommendation.category.value, "process_improvement")
    
    def _create_phase_tasks(
        self,
        phase_name: str,
        task_template: Dict[str, Any],
        recommendation: Recommendation,
        context: PlanningContext
    ) -> List[ImplementationTask]:
        """Create tasks for a specific phase."""
        tasks = []
        phase_tasks = task_template.get(phase_name.lower().replace(" ", "_"), {})
        
        if not phase_tasks:
            # Create default task for phase
            task = ImplementationTask(
                id=f"{phase_name.lower()}_task",
                title=f"{phase_name}",
                description=f"Complete {phase_name.lower()} activities for {recommendation.title}",
                task_type=self._get_task_type(phase_name),
                priority=self._get_task_priority(recommendation.priority.value),
                estimated_duration_days=7,
                dependencies=[],
                deliverables=[f"{phase_name} deliverables"]
            )
            tasks.append(task)
        else:
            # Create tasks based on template
            for task_key, task_config in phase_tasks.items():
                task = ImplementationTask(
                    id=f"{phase_name.lower()}_{task_key}",
                    title=task_config["title"],
                    description=task_config["description"],
                    task_type=self._get_task_type(phase_name),
                    priority=self._get_task_priority(recommendation.priority.value),
                    estimated_duration_days=task_config["duration"],
                    dependencies=task_config.get("dependencies", []),
                    deliverables=task_config.get("deliverables", []),
                    resources_required=self._get_required_resources(task_config, context)
                )
                tasks.append(task)
        
        return tasks
    
    def _get_task_type(self, phase_name: str) -> TaskType:
        """Get task type based on phase name."""
        type_mapping = {
            "Planning": TaskType.PLANNING,
            "Analysis": TaskType.ANALYSIS,
            "Research": TaskType.ANALYSIS,
            "Design": TaskType.PLANNING,
            "Development": TaskType.DEVELOPMENT,
            "Implementation": TaskType.DEVELOPMENT,
            "Testing": TaskType.TESTING,
            "Deployment": TaskType.DEPLOYMENT,
            "Training": TaskType.TRAINING,
            "Monitoring": TaskType.REVIEW,
            "Review": TaskType.REVIEW
        }
        
        return type_mapping.get(phase_name, TaskType.PLANNING)
    
    def _get_task_priority(self, recommendation_priority: str) -> TaskPriority:
        """Get task priority based on recommendation priority."""
        priority_mapping = {
            "critical": TaskPriority.CRITICAL,
            "high": TaskPriority.HIGH,
            "medium": TaskPriority.MEDIUM,
            "low": TaskPriority.LOW
        }
        
        return priority_mapping.get(recommendation_priority, TaskPriority.MEDIUM)
    
    def _get_required_resources(
        self,
        task_config: Dict[str, Any],
        context: PlanningContext
    ) -> List[str]:
        """Get required resources for a task."""
        resources = []
        
        # Add basic resources based on task type
        if "development" in task_config.get("title", "").lower():
            resources.extend(["Development environment", "Technical team"])
        elif "testing" in task_config.get("title", "").lower():
            resources.extend(["Testing environment", "QA team"])
        elif "training" in task_config.get("title", "").lower():
            resources.extend(["Training materials", "Training facilities"])
        
        # Add resources based on context
        if context.available_resources:
            resources.extend(list(context.available_resources.keys()))
        
        return list(set(resources))  # Remove duplicates
    
    def _estimate_total_cost(
        self,
        recommendation: Recommendation,
        phases: List[ImplementationPhase]
    ) -> Optional[float]:
        """Estimate total implementation cost."""
        if recommendation.cost_estimate:
            return recommendation.cost_estimate
        
        # Estimate based on duration and complexity
        total_days = sum(len(phase.tasks) * 7 for phase in phases)  # Rough estimate
        daily_rate = 500  # Default daily rate
        return total_days * daily_rate
    
    def _identify_risk_factors(
        self,
        recommendation: Recommendation,
        context: PlanningContext
    ) -> List[str]:
        """Identify potential risk factors for implementation."""
        risks = []
        
        # Add risks based on recommendation type
        if recommendation.category.value == "technology_adoption":
            risks.extend([
                "Technology compatibility issues",
                "Integration challenges",
                "User adoption resistance"
            ])
        elif recommendation.category.value == "process_improvement":
            risks.extend([
                "Process disruption during transition",
                "Staff resistance to change",
                "Inadequate training"
            ])
        elif recommendation.category.value == "risk_management":
            risks.extend([
                "Incomplete risk assessment",
                "Inadequate mitigation measures",
                "Monitoring gaps"
            ])
        
        # Add context-specific risks
        if context.budget_constraints and recommendation.cost_estimate:
            if recommendation.cost_estimate > context.budget_constraints:
                risks.append("Budget overrun")
        
        if context.timeline_constraints:
            if recommendation.time_to_implement == "6-12 months":
                risks.append("Timeline pressure")
        
        return risks
    
    def _define_success_criteria(self, recommendation: Recommendation) -> List[str]:
        """Define success criteria for the implementation."""
        criteria = []
        
        # Add criteria based on expected impact
        if recommendation.expected_impact:
            for impact_type, impact_level in recommendation.expected_impact.items():
                if impact_level in ["high", "improved", "enhanced"]:
                    criteria.append(f"Achieve {impact_level} {impact_type}")
        
        # Add category-specific criteria
        if recommendation.category.value == "technology_adoption":
            criteria.extend([
                "Successful technology deployment",
                "User adoption rate > 80%",
                "Performance improvement achieved"
            ])
        elif recommendation.category.value == "process_improvement":
            criteria.extend([
                "Process efficiency improved by 20%",
                "Error rate reduced by 50%",
                "Staff satisfaction increased"
            ])
        elif recommendation.category.value == "risk_management":
            criteria.extend([
                "Risk exposure reduced by 30%",
                "Compliance requirements met",
                "Risk monitoring system operational"
            ])
        
        return criteria
    
    async def update_plan_progress(
        self,
        plan: ImplementationPlan,
        task_updates: Dict[str, Dict[str, Any]]
    ) -> ImplementationPlan:
        """Update implementation plan with progress information."""
        try:
            for phase in plan.phases:
                for task in phase.tasks:
                    if task.id in task_updates:
                        update = task_updates[task.id]
                        
                        # Update task status and progress
                        if "status" in update:
                            task.status = TaskStatus(update["status"])
                        if "progress_percentage" in update:
                            task.progress_percentage = update["progress_percentage"]
                        if "notes" in update:
                            task.notes = update["notes"]
                        
                        # Update dates if task is started/completed
                        if task.status == TaskStatus.IN_PROGRESS and not task.start_date:
                            task.start_date = datetime.now()
                        elif task.status == TaskStatus.COMPLETED and not task.end_date:
                            task.end_date = datetime.now()
                
                # Update phase progress
                phase.progress_percentage = self._calculate_phase_progress(phase)
                phase.status = self._determine_phase_status(phase)
            
            plan.last_updated = datetime.now()
            logger.info(f"Updated implementation plan progress for: {plan.recommendation_title}")
            return plan
            
        except Exception as e:
            logger.error(f"Error updating plan progress: {e}")
            return plan
    
    def _calculate_phase_progress(self, phase: ImplementationPhase) -> float:
        """Calculate overall progress for a phase."""
        if not phase.tasks:
            return 0.0
        
        total_progress = sum(task.progress_percentage for task in phase.tasks)
        return total_progress / len(phase.tasks)
    
    def _determine_phase_status(self, phase: ImplementationPhase) -> TaskStatus:
        """Determine overall status for a phase."""
        if not phase.tasks:
            return TaskStatus.NOT_STARTED
        
        completed_tasks = sum(1 for task in phase.tasks if task.status == TaskStatus.COMPLETED)
        total_tasks = len(phase.tasks)
        
        if completed_tasks == total_tasks:
            return TaskStatus.COMPLETED
        elif completed_tasks > 0:
            return TaskStatus.IN_PROGRESS
        else:
            return TaskStatus.NOT_STARTED
    
    async def get_implementation_report(self, plan: ImplementationPlan) -> Dict[str, Any]:
        """Generate a comprehensive implementation report."""
        total_tasks = sum(len(phase.tasks) for phase in plan.phases)
        completed_tasks = sum(
            sum(1 for task in phase.tasks if task.status == TaskStatus.COMPLETED)
            for phase in plan.phases
        )
        
        report = {
            "plan_summary": {
                "recommendation_title": plan.recommendation_title,
                "total_phases": len(plan.phases),
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "overall_progress": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
                "total_duration_days": plan.total_duration_days,
                "estimated_cost": plan.total_estimated_cost
            },
            "phase_progress": [
                {
                    "phase_name": phase.name,
                    "status": phase.status.value,
                    "progress_percentage": phase.progress_percentage,
                    "tasks_completed": sum(1 for task in phase.tasks if task.status == TaskStatus.COMPLETED),
                    "total_tasks": len(phase.tasks)
                }
                for phase in plan.phases
            ],
            "risk_factors": plan.risk_factors,
            "success_criteria": plan.success_criteria,
            "next_steps": self._identify_next_steps(plan)
        }
        
        return report
    
    def _identify_next_steps(self, plan: ImplementationPlan) -> List[str]:
        """Identify next steps for implementation."""
        next_steps = []
        
        for phase in plan.phases:
            if phase.status == TaskStatus.NOT_STARTED:
                next_steps.append(f"Start {phase.name} phase")
                break
            elif phase.status == TaskStatus.IN_PROGRESS:
                incomplete_tasks = [task for task in phase.tasks if task.status != TaskStatus.COMPLETED]
                if incomplete_tasks:
                    next_steps.append(f"Complete {incomplete_tasks[0].title}")
                break
        
        return next_steps
