"""
Dashboard Exporter

Provides interactive dashboard export capabilities including:
- Dashboard state export
- Interactive chart export
- Configuration export
- Dashboard templates
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DashboardExporter:
    """Interactive dashboard export system."""
    
    def __init__(self, output_dir: str = "dashboard_exports"):
        """Initialize the dashboard exporter.
        
        Args:
            output_dir: Directory to save exported dashboards
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def export_dashboard_state(self, 
                              dashboard_config: Dict[str, Any],
                              charts_data: List[Dict[str, Any]],
                              filename: Optional[str] = None) -> Dict[str, Any]:
        """Export dashboard state and configuration.
        
        Args:
            dashboard_config: Dashboard configuration
            charts_data: Chart data and configurations
            filename: Optional custom filename
            
        Returns:
            Export result with file information
        """
        try:
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"dashboard_export_{timestamp}.json"
            
            # Create export data
            export_data = {
                "dashboard_config": dashboard_config,
                "charts_data": charts_data,
                "export_metadata": {
                    "exported_at": datetime.now().isoformat(),
                    "version": "1.0",
                    "type": "dashboard_export"
                }
            }
            
            # Save export
            filepath = self.output_dir / filename
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            return {
                "success": True,
                "filename": filename,
                "filepath": str(filepath),
                "size": filepath.stat().st_size,
                "message": f"Dashboard exported successfully to {filename}"
            }
            
        except Exception as e:
            logger.error(f"Error exporting dashboard: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to export dashboard"
            }
    
    def export_chart_config(self, 
                           chart_config: Dict[str, Any],
                           chart_data: Dict[str, Any],
                           filename: Optional[str] = None) -> Dict[str, Any]:
        """Export individual chart configuration and data.
        
        Args:
            chart_config: Chart configuration
            chart_data: Chart data
            filename: Optional custom filename
            
        Returns:
            Export result with file information
        """
        try:
            # Generate filename if not provided
            if not filename:
                chart_name = chart_config.get("name", "chart")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{chart_name}_{timestamp}.json"
            
            # Create export data
            export_data = {
                "chart_config": chart_config,
                "chart_data": chart_data,
                "export_metadata": {
                    "exported_at": datetime.now().isoformat(),
                    "version": "1.0",
                    "type": "chart_export"
                }
            }
            
            # Save export
            filepath = self.output_dir / filename
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            return {
                "success": True,
                "filename": filename,
                "filepath": str(filepath),
                "size": filepath.stat().st_size,
                "message": f"Chart exported successfully to {filename}"
            }
            
        except Exception as e:
            logger.error(f"Error exporting chart: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to export chart"
            }
    
    def create_dashboard_template(self, 
                                template_name: str,
                                template_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a reusable dashboard template.
        
        Args:
            template_name: Name of the template
            template_config: Template configuration
            
        Returns:
            Template creation result
        """
        try:
            # Create template data
            template_data = {
                "template_name": template_name,
                "template_config": template_config,
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
                "type": "dashboard_template"
            }
            
            # Save template
            filename = f"{template_name}_template.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(template_data, f, indent=2, default=str)
            
            return {
                "success": True,
                "template_name": template_name,
                "filename": filename,
                "filepath": str(filepath),
                "message": f"Dashboard template created: {template_name}"
            }
            
        except Exception as e:
            logger.error(f"Error creating dashboard template: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to create dashboard template"
            }
    
    def list_dashboard_exports(self) -> List[Dict[str, Any]]:
        """List all dashboard exports.
        
        Returns:
            List of export metadata
        """
        exports = []
        for filepath in self.output_dir.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    export_type = data.get("export_metadata", {}).get("type", "unknown")
                    
                    exports.append({
                        "filename": filepath.name,
                        "path": str(filepath),
                        "type": export_type,
                        "size": filepath.stat().st_size,
                        "created": datetime.fromtimestamp(filepath.stat().st_ctime).isoformat()
                    })
            except Exception as e:
                logger.error(f"Error reading export {filepath}: {e}")
        
        return exports
    
    def get_dashboard_export(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get a specific dashboard export by filename.
        
        Args:
            filename: Name of the export file
            
        Returns:
            Export data or None if not found
        """
        filepath = self.output_dir / filename
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading dashboard export {filename}: {e}")
            return None
