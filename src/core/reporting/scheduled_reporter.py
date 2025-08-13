"""
Scheduled Reporter

Provides automated report delivery capabilities including:
- Scheduled report generation
- Email delivery
- Report scheduling
- Delivery tracking
"""

import json
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
import threading

logger = logging.getLogger(__name__)


class ScheduledReporter:
    """Automated scheduled reporting system."""
    
    def __init__(self, config_dir: str = "scheduled_reports"):
        """Initialize the scheduled reporter.
        
        Args:
            config_dir: Directory to store scheduling configurations
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Active schedules
        self.active_schedules = {}
        
        # Schedule thread
        self.scheduler_thread = None
        self.running = False
    
    def create_schedule(self, 
                       schedule_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new scheduled report.
        
        Args:
            schedule_config: Schedule configuration
            
        Returns:
            Schedule creation result
        """
        try:
            schedule_id = schedule_config.get("schedule_id")
            if not schedule_id:
                schedule_id = f"schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                schedule_config["schedule_id"] = schedule_id
            
            # Validate schedule configuration
            validation_result = self._validate_schedule_config(schedule_config)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": validation_result["error"],
                    "message": "Invalid schedule configuration"
                }
            
            # Save schedule configuration
            filename = f"{schedule_id}_config.json"
            filepath = self.config_dir / filename
            
            schedule_config.update({
                "created_at": datetime.now().isoformat(),
                "status": "active"
            })
            
            with open(filepath, 'w') as f:
                json.dump(schedule_config, f, indent=2, default=str)
            
            # Add to active schedules
            self.active_schedules[schedule_id] = schedule_config
            
            # Schedule the job
            self._schedule_job(schedule_config)
            
            return {
                "success": True,
                "schedule_id": schedule_id,
                "filename": filename,
                "message": f"Scheduled report created: {schedule_id}"
            }
            
        except Exception as e:
            logger.error(f"Error creating schedule: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to create scheduled report"
            }
    
    def _validate_schedule_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate schedule configuration.
        
        Args:
            config: Schedule configuration to validate
            
        Returns:
            Validation result
        """
        required_fields = ["report_type", "frequency", "recipients"]
        
        for field in required_fields:
            if field not in config:
                return {
                    "valid": False,
                    "error": f"Missing required field: {field}"
                }
        
        # Validate frequency
        valid_frequencies = ["daily", "weekly", "monthly", "custom"]
        if config["frequency"] not in valid_frequencies:
            return {
                "valid": False,
                "error": f"Invalid frequency: {config['frequency']}"
            }
        
        # Validate recipients
        if not isinstance(config["recipients"], list) or not config["recipients"]:
            return {
                "valid": False,
                "error": "Recipients must be a non-empty list"
            }
        
        return {"valid": True}
    
    def _schedule_job(self, schedule_config: Dict[str, Any]):
        """Schedule a job based on configuration.
        
        Args:
            schedule_config: Schedule configuration
        """
        schedule_id = schedule_config["schedule_id"]
        frequency = schedule_config["frequency"]
        
        if frequency == "daily":
            schedule.every().day.at("09:00").do(
                self._execute_scheduled_report, schedule_config
            ).tag(schedule_id)
        elif frequency == "weekly":
            schedule.every().monday.at("09:00").do(
                self._execute_scheduled_report, schedule_config
            ).tag(schedule_id)
        elif frequency == "monthly":
            schedule.every().month.at("09:00").do(
                self._execute_scheduled_report, schedule_config
            ).tag(schedule_id)
        elif frequency == "custom":
            # Custom scheduling logic
            custom_time = schedule_config.get("custom_time", "09:00")
            custom_day = schedule_config.get("custom_day", "monday")
            
            if custom_day == "monday":
                schedule.every().monday.at(custom_time).do(
                    self._execute_scheduled_report, schedule_config
                ).tag(schedule_id)
            elif custom_day == "tuesday":
                schedule.every().tuesday.at(custom_time).do(
                    self._execute_scheduled_report, schedule_config
                ).tag(schedule_id)
            # Add more days as needed
    
    def _execute_scheduled_report(self, schedule_config: Dict[str, Any]):
        """Execute a scheduled report.
        
        Args:
            schedule_config: Schedule configuration
        """
        try:
            schedule_id = schedule_config["schedule_id"]
            report_type = schedule_config["report_type"]
            recipients = schedule_config["recipients"]
            
            logger.info(f"Executing scheduled report: {schedule_id}")
            
            # Generate report (placeholder - would integrate with ReportGenerator)
            report_result = self._generate_report(report_type, schedule_config)
            
            if report_result["success"]:
                # Send report to recipients
                delivery_result = self._deliver_report(
                    report_result["report"], recipients, schedule_config
                )
                
                # Log execution
                self._log_execution(schedule_id, report_result, delivery_result)
                
                logger.info(f"Scheduled report completed: {schedule_id}")
            else:
                logger.error(f"Scheduled report failed: {schedule_id}")
                
        except Exception as e:
            logger.error(f"Error executing scheduled report: {e}")
    
    def _generate_report(self, report_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a report (placeholder implementation).
        
        Args:
            report_type: Type of report to generate
            config: Report configuration
            
        Returns:
            Report generation result
        """
        # This would integrate with the actual ReportGenerator
        return {
            "success": True,
            "report": {
                "type": report_type,
                "content": f"Generated {report_type} report",
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _deliver_report(self, report: Dict[str, Any], 
                       recipients: List[str], 
                       config: Dict[str, Any]) -> Dict[str, Any]:
        """Deliver report to recipients (placeholder implementation).
        
        Args:
            report: Report to deliver
            recipients: List of recipient emails
            config: Delivery configuration
            
        Returns:
            Delivery result
        """
        # This would integrate with email delivery system
        return {
            "success": True,
            "delivered_to": recipients,
            "timestamp": datetime.now().isoformat()
        }
    
    def _log_execution(self, schedule_id: str, 
                      report_result: Dict[str, Any], 
                      delivery_result: Dict[str, Any]):
        """Log report execution.
        
        Args:
            schedule_id: Schedule ID
            report_result: Report generation result
            delivery_result: Delivery result
        """
        log_entry = {
            "schedule_id": schedule_id,
            "executed_at": datetime.now().isoformat(),
            "report_result": report_result,
            "delivery_result": delivery_result
        }
        
        # Save execution log
        log_filename = f"{schedule_id}_execution_log.json"
        log_filepath = self.config_dir / log_filename
        
        try:
            with open(log_filepath, 'w') as f:
                json.dump(log_entry, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error logging execution: {e}")
    
    def list_schedules(self) -> List[Dict[str, Any]]:
        """List all scheduled reports.
        
        Returns:
            List of schedule configurations
        """
        schedules = []
        for filepath in self.config_dir.glob("*_config.json"):
            try:
                with open(filepath, 'r') as f:
                    schedule_config = json.load(f)
                    schedules.append({
                        "schedule_id": schedule_config.get("schedule_id"),
                        "report_type": schedule_config.get("report_type"),
                        "frequency": schedule_config.get("frequency"),
                        "status": schedule_config.get("status", "unknown"),
                        "created_at": schedule_config.get("created_at"),
                        "filename": filepath.name
                    })
            except Exception as e:
                logger.error(f"Error reading schedule {filepath}: {e}")
        
        return schedules
    
    def get_schedule(self, schedule_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific schedule by ID.
        
        Args:
            schedule_id: Schedule ID
            
        Returns:
            Schedule configuration or None if not found
        """
        filename = f"{schedule_id}_config.json"
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading schedule {schedule_id}: {e}")
            return None
    
    def update_schedule(self, schedule_id: str, 
                       updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update a scheduled report.
        
        Args:
            schedule_id: Schedule ID
            updates: Updates to apply
            
        Returns:
            Update result
        """
        try:
            # Get current schedule
            current_schedule = self.get_schedule(schedule_id)
            if not current_schedule:
                return {
                    "success": False,
                    "error": f"Schedule not found: {schedule_id}",
                    "message": "Schedule does not exist"
                }
            
            # Apply updates
            current_schedule.update(updates)
            current_schedule["updated_at"] = datetime.now().isoformat()
            
            # Validate updated configuration
            validation_result = self._validate_schedule_config(current_schedule)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": validation_result["error"],
                    "message": "Invalid updated configuration"
                }
            
            # Save updated configuration
            filename = f"{schedule_id}_config.json"
            filepath = self.config_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(current_schedule, f, indent=2, default=str)
            
            # Update active schedules
            self.active_schedules[schedule_id] = current_schedule
            
            # Reschedule job
            schedule.clear(schedule_id)
            self._schedule_job(current_schedule)
            
            return {
                "success": True,
                "schedule_id": schedule_id,
                "message": f"Schedule updated: {schedule_id}"
            }
            
        except Exception as e:
            logger.error(f"Error updating schedule: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to update schedule"
            }
    
    def delete_schedule(self, schedule_id: str) -> Dict[str, Any]:
        """Delete a scheduled report.
        
        Args:
            schedule_id: Schedule ID
            
        Returns:
            Deletion result
        """
        try:
            # Remove from active schedules
            if schedule_id in self.active_schedules:
                del self.active_schedules[schedule_id]
            
            # Clear scheduled job
            schedule.clear(schedule_id)
            
            # Delete configuration file
            filename = f"{schedule_id}_config.json"
            filepath = self.config_dir / filename
            
            if filepath.exists():
                filepath.unlink()
            
            return {
                "success": True,
                "schedule_id": schedule_id,
                "message": f"Schedule deleted: {schedule_id}"
            }
            
        except Exception as e:
            logger.error(f"Error deleting schedule: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to delete schedule"
            }
    
    def start_scheduler(self):
        """Start the scheduler thread."""
        if not self.running:
            self.running = True
            self.scheduler_thread = threading.Thread(target=self._run_scheduler)
            self.scheduler_thread.daemon = True
            self.scheduler_thread.start()
            logger.info("Scheduled reporter started")
    
    def stop_scheduler(self):
        """Stop the scheduler thread."""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
        logger.info("Scheduled reporter stopped")
    
    def _run_scheduler(self):
        """Run the scheduler loop."""
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
