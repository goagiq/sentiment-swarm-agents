"""
Analytics Workflow Management

Workflow orchestration and management for analytics dashboard with:
- Workflow orchestration
- Scheduled analytics
- Collaborative analytics
- Version control
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any


class AnalyticsWorkflow:
    """
    Analytics workflow management component.
    """
    
    def __init__(self):
        """Initialize the analytics workflow manager."""
        self.api_base_url = "http://localhost:8003"
        self.workflows = {}
        self.scheduled_tasks = {}
        self.collaborators = {}
        
    def create_workflow(self, name: str, steps: List[Dict[str, Any]], 
                       config: Optional[Dict[str, Any]] = None) -> str:
        """Create a new analytics workflow."""
        workflow_id = f"workflow_{len(self.workflows) + 1}"
        
        workflow = {
            'id': workflow_id,
            'name': name,
            'steps': steps,
            'config': config or {},
            'created_at': datetime.now().isoformat(),
            'status': 'draft',
            'version': '1.0',
            'collaborators': []
        }
        
        self.workflows[workflow_id] = workflow
        return workflow_id
    
    def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute an analytics workflow."""
        if workflow_id not in self.workflows:
            return {'error': 'Workflow not found'}
        
        workflow = self.workflows[workflow_id]
        workflow['status'] = 'running'
        workflow['started_at'] = datetime.now().isoformat()
        
        results = []
        
        try:
            for step in workflow['steps']:
                step_result = self._execute_step(step)
                results.append(step_result)
                
                if step_result.get('status') == 'failed':
                    workflow['status'] = 'failed'
                    break
            
            workflow['status'] = 'completed'
            workflow['completed_at'] = datetime.now().isoformat()
            workflow['results'] = results
            
            return {
                'workflow_id': workflow_id,
                'status': workflow['status'],
                'results': results
            }
            
        except Exception as e:
            workflow['status'] = 'failed'
            workflow['error'] = str(e)
            return {
                'workflow_id': workflow_id,
                'status': 'failed',
                'error': str(e)
            }
    
    def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step."""
        step_type = step.get('type', 'unknown')
        
        if step_type == 'data_processing':
            return self._execute_data_processing_step(step)
        elif step_type == 'analytics':
            return self._execute_analytics_step(step)
        elif step_type == 'visualization':
            return self._execute_visualization_step(step)
        elif step_type == 'reporting':
            return self._execute_reporting_step(step)
        else:
            return {
                'status': 'failed',
                'error': f'Unknown step type: {step_type}'
            }
    
    def _execute_data_processing_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a data processing step."""
        try:
            # This would integrate with the data processing pipeline
            return {
                'step_type': 'data_processing',
                'status': 'completed',
                'output': f'Processed {step.get("data_source", "unknown")}',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'step_type': 'data_processing',
                'status': 'failed',
                'error': str(e)
            }
    
    def _execute_analytics_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an analytics step."""
        try:
            # This would integrate with the analytics engine
            return {
                'step_type': 'analytics',
                'status': 'completed',
                'output': f'Analytics completed for {step.get("analysis_type", "unknown")}',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'step_type': 'analytics',
                'status': 'failed',
                'error': str(e)
            }
    
    def _execute_visualization_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a visualization step."""
        try:
            # This would integrate with the visualization engine
            return {
                'step_type': 'visualization',
                'status': 'completed',
                'output': f'Visualization created for {step.get("chart_type", "unknown")}',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'step_type': 'visualization',
                'status': 'failed',
                'error': str(e)
            }
    
    def _execute_reporting_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a reporting step."""
        try:
            # This would integrate with the reporting engine
            return {
                'step_type': 'reporting',
                'status': 'completed',
                'output': f'Report generated for {step.get("report_type", "unknown")}',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'step_type': 'reporting',
                'status': 'failed',
                'error': str(e)
            }
    
    def schedule_workflow(self, workflow_id: str, schedule: Dict[str, Any]) -> str:
        """Schedule a workflow for execution."""
        if workflow_id not in self.workflows:
            return None
        
        task_id = f"task_{len(self.scheduled_tasks) + 1}"
        
        scheduled_task = {
            'id': task_id,
            'workflow_id': workflow_id,
            'schedule': schedule,
            'status': 'scheduled',
            'created_at': datetime.now().isoformat(),
            'next_run': self._calculate_next_run(schedule)
        }
        
        self.scheduled_tasks[task_id] = scheduled_task
        return task_id
    
    def _calculate_next_run(self, schedule: Dict[str, Any]) -> str:
        """Calculate the next run time based on schedule."""
        schedule_type = schedule.get('type', 'once')
        
        if schedule_type == 'once':
            return schedule.get('datetime', datetime.now().isoformat())
        elif schedule_type == 'daily':
            next_run = datetime.now() + timedelta(days=1)
            return next_run.replace(
                hour=schedule.get('hour', 0),
                minute=schedule.get('minute', 0)
            ).isoformat()
        elif schedule_type == 'weekly':
            days_ahead = schedule.get('day_of_week', 0) - datetime.now().weekday()
            if days_ahead <= 0:
                days_ahead += 7
            next_run = datetime.now() + timedelta(days=days_ahead)
            return next_run.replace(
                hour=schedule.get('hour', 0),
                minute=schedule.get('minute', 0)
            ).isoformat()
        else:
            return datetime.now().isoformat()
    
    def add_collaborator(self, workflow_id: str, user_id: str, 
                        permissions: List[str]) -> bool:
        """Add a collaborator to a workflow."""
        if workflow_id not in self.workflows:
            return False
        
        collaborator = {
            'user_id': user_id,
            'permissions': permissions,
            'added_at': datetime.now().isoformat()
        }
        
        self.workflows[workflow_id]['collaborators'].append(collaborator)
        return True
    
    def version_workflow(self, workflow_id: str) -> str:
        """Create a new version of a workflow."""
        if workflow_id not in self.workflows:
            return None
        
        original_workflow = self.workflows[workflow_id]
        new_version = str(float(original_workflow['version']) + 0.1)
        
        new_workflow_id = f"{workflow_id}_v{new_version}"
        
        new_workflow = original_workflow.copy()
        new_workflow['id'] = new_workflow_id
        new_workflow['version'] = new_version
        new_workflow['created_at'] = datetime.now().isoformat()
        new_workflow['status'] = 'draft'
        new_workflow['parent_version'] = workflow_id
        
        self.workflows[new_workflow_id] = new_workflow
        return new_workflow_id
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get the status of a workflow."""
        if workflow_id not in self.workflows:
            return {'error': 'Workflow not found'}
        
        workflow = self.workflows[workflow_id]
        return {
            'id': workflow['id'],
            'name': workflow['name'],
            'status': workflow['status'],
            'version': workflow['version'],
            'created_at': workflow['created_at'],
            'started_at': workflow.get('started_at'),
            'completed_at': workflow.get('completed_at'),
            'error': workflow.get('error')
        }
    
    def get_scheduled_tasks(self) -> List[Dict[str, Any]]:
        """Get all scheduled tasks."""
        return list(self.scheduled_tasks.values())
    
    def render_workflow_manager(self):
        """Render the workflow management interface."""
        st.markdown("## 🔄 Analytics Workflow Manager")
        
        # Create new workflow
        with st.expander("Create New Workflow", expanded=False):
            self._render_create_workflow_form()
        
        # Workflow list
        st.markdown("### 📋 Workflows")
        self._render_workflow_list()
        
        # Scheduled tasks
        st.markdown("### ⏰ Scheduled Tasks")
        self._render_scheduled_tasks()
        
        # Collaboration
        st.markdown("### 👥 Collaboration")
        self._render_collaboration_panel()
    
    def _render_create_workflow_form(self):
        """Render the create workflow form."""
        with st.form("create_workflow"):
            workflow_name = st.text_input("Workflow Name")
            
            st.markdown("#### Workflow Steps")
            
            # Step 1: Data Processing
            st.markdown("**Step 1: Data Processing**")
            data_source = st.selectbox(
                "Data Source",
                ["API", "Database", "File", "Stream"],
                key="step1_source"
            )
            
            # Step 2: Analytics
            st.markdown("**Step 2: Analytics**")
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Descriptive", "Predictive", "Diagnostic", "Prescriptive"],
                key="step2_analysis"
            )
            
            # Step 3: Visualization
            st.markdown("**Step 3: Visualization**")
            chart_type = st.selectbox(
                "Chart Type",
                ["Line Chart", "Bar Chart", "Scatter Plot", "Heatmap"],
                key="step3_chart"
            )
            
            # Step 4: Reporting
            st.markdown("**Step 4: Reporting**")
            report_type = st.selectbox(
                "Report Type",
                ["Summary", "Detailed", "Executive", "Technical"],
                key="step4_report"
            )
            
            if st.form_submit_button("Create Workflow"):
                if workflow_name:
                    steps = [
                        {
                            'type': 'data_processing',
                            'data_source': data_source
                        },
                        {
                            'type': 'analytics',
                            'analysis_type': analysis_type
                        },
                        {
                            'type': 'visualization',
                            'chart_type': chart_type
                        },
                        {
                            'type': 'reporting',
                            'report_type': report_type
                        }
                    ]
                    
                    workflow_id = self.create_workflow(workflow_name, steps)
                    st.success(f"Workflow created with ID: {workflow_id}")
                else:
                    st.error("Please enter a workflow name")
    
    def _render_workflow_list(self):
        """Render the workflow list."""
        if not self.workflows:
            st.info("No workflows created yet")
            return
        
        for workflow_id, workflow in self.workflows.items():
            with st.expander(f"{workflow['name']} (v{workflow['version']})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Status:** {workflow['status']}")
                    st.write(f"**Created:** {workflow['created_at'][:10]}")
                
                with col2:
                    if st.button("Execute", key=f"exec_{workflow_id}"):
                        result = self.execute_workflow(workflow_id)
                        if result.get('status') == 'completed':
                            st.success("Workflow executed successfully")
                        else:
                            st.error(f"Workflow failed: {result.get('error')}")
                
                with col3:
                    if st.button("Schedule", key=f"sched_{workflow_id}"):
                        self._show_schedule_dialog(workflow_id)
                
                # Show workflow steps
                st.markdown("**Steps:**")
                for i, step in enumerate(workflow['steps'], 1):
                    st.write(f"{i}. {step['type']}: {step.get('data_source', step.get('analysis_type', step.get('chart_type', step.get('report_type', 'Unknown'))))}")
    
    def _show_schedule_dialog(self, workflow_id: str):
        """Show the schedule dialog."""
        st.markdown("#### Schedule Workflow")
        
        schedule_type = st.selectbox(
            "Schedule Type",
            ["once", "daily", "weekly"],
            key=f"schedule_type_{workflow_id}"
        )
        
        if schedule_type == "once":
            schedule_datetime = st.datetime_input(
                "Execution Date/Time",
                key=f"schedule_datetime_{workflow_id}"
            )
            schedule = {
                'type': 'once',
                'datetime': schedule_datetime.isoformat()
            }
        elif schedule_type == "daily":
            col1, col2 = st.columns(2)
            with col1:
                hour = st.number_input("Hour", 0, 23, 9, key=f"schedule_hour_{workflow_id}")
            with col2:
                minute = st.number_input("Minute", 0, 59, 0, key=f"schedule_minute_{workflow_id}")
            schedule = {
                'type': 'daily',
                'hour': hour,
                'minute': minute
            }
        else:  # weekly
            col1, col2, col3 = st.columns(3)
            with col1:
                day_of_week = st.selectbox(
                    "Day of Week",
                    range(7),
                    format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
                    key=f"schedule_day_{workflow_id}"
                )
            with col2:
                hour = st.number_input("Hour", 0, 23, 9, key=f"schedule_hour_w_{workflow_id}")
            with col3:
                minute = st.number_input("Minute", 0, 59, 0, key=f"schedule_minute_w_{workflow_id}")
            schedule = {
                'type': 'weekly',
                'day_of_week': day_of_week,
                'hour': hour,
                'minute': minute
            }
        
        if st.button("Schedule", key=f"confirm_schedule_{workflow_id}"):
            task_id = self.schedule_workflow(workflow_id, schedule)
            if task_id:
                st.success(f"Workflow scheduled with task ID: {task_id}")
    
    def _render_scheduled_tasks(self):
        """Render the scheduled tasks list."""
        tasks = self.get_scheduled_tasks()
        
        if not tasks:
            st.info("No scheduled tasks")
            return
        
        for task in tasks:
            workflow = self.workflows.get(task['workflow_id'], {})
            st.write(f"**{workflow.get('name', 'Unknown')}** - {task['status']} - Next run: {task['next_run'][:16]}")
    
    def _render_collaboration_panel(self):
        """Render the collaboration panel."""
        st.markdown("#### Add Collaborator")
        
        workflow_id = st.selectbox(
            "Select Workflow",
            list(self.workflows.keys()),
            format_func=lambda x: self.workflows[x]['name'],
            key="collab_workflow"
        )
        
        if workflow_id:
            user_id = st.text_input("User ID", key="collab_user")
            permissions = st.multiselect(
                "Permissions",
                ["read", "write", "execute", "admin"],
                key="collab_perms"
            )
            
            if st.button("Add Collaborator"):
                if user_id and permissions:
                    success = self.add_collaborator(workflow_id, user_id, permissions)
                    if success:
                        st.success(f"Collaborator {user_id} added")
                    else:
                        st.error("Failed to add collaborator")
                else:
                    st.error("Please enter user ID and select permissions")


# Factory function for creating analytics workflow
def create_analytics_workflow() -> AnalyticsWorkflow:
    """Create an analytics workflow manager."""
    return AnalyticsWorkflow()
