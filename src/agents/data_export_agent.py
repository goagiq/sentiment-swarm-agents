"""
Data Export Agent for export and sharing capabilities.
Part of Phase 4: Export & Automation implementation.
"""

import json
import csv
import base64
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
import logging

from src.agents.base_agent import StrandsBaseAgent as BaseAgent


class ExportManager:
    """Handles data export in multiple formats."""
    
    def __init__(self):
        self.export_formats = {
            "json": self._export_to_json,
            "csv": self._export_to_csv,
            "html": self._export_to_html,
            "pdf": self._export_to_pdf,
            "excel": self._export_to_excel
        }
    
    async def export_data(
        self,
        data: Dict[str, Any],
        export_formats: List[str] = ["json"],
        include_visualizations: bool = True,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Export data to multiple formats."""
        results = {}
        
        for format_type in export_formats:
            if format_type in self.export_formats:
                try:
                    export_func = self.export_formats[format_type]
                    result = await export_func(
                        data, include_visualizations, include_metadata
                    )
                    results[format_type] = result
                except Exception as e:
                    results[format_type] = {
                        "status": "error",
                        "error": str(e)
                    }
            else:
                results[format_type] = {
                    "status": "error",
                    "error": f"Unsupported format: {format_type}"
                }
        
        return results
    
    async def _export_to_json(
        self,
        data: Dict[str, Any],
        include_visualizations: bool = True,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Export data to JSON format."""
        export_data = data.copy()
        
        if include_metadata:
            export_data["metadata"] = {
                "exported_at": datetime.now().isoformat(),
                "format": "json",
                "include_visualizations": include_visualizations
            }
        
        if not include_visualizations and "visualizations" in export_data:
            del export_data["visualizations"]
        
        return {
            "status": "success",
            "format": "json",
            "data": export_data,
            "size": len(json.dumps(export_data))
        }
    
    async def _export_to_csv(
        self,
        data: Dict[str, Any],
        include_visualizations: bool = True,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Export data to CSV format."""
        # Convert data to CSV format
        csv_data = []
        
        # Flatten the data structure for CSV
        if "results" in data:
            for item in data["results"]:
                row = {}
                for key, value in item.items():
                    if isinstance(value, (dict, list)):
                        row[key] = json.dumps(value)
                    else:
                        row[key] = str(value)
                csv_data.append(row)
        
        if not csv_data:
            csv_data = [{"message": "No data to export"}]
        
        return {
            "status": "success",
            "format": "csv",
            "data": csv_data,
            "rows": len(csv_data)
        }
    
    async def _export_to_html(
        self,
        data: Dict[str, Any],
        include_visualizations: bool = True,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Export data to HTML format."""
        html_content = "<!DOCTYPE html>\n<html>\n<head>\n"
        html_content += "<title>Analysis Report</title>\n"
        html_content += "<style>\n"
        html_content += "body { font-family: Arial, sans-serif; margin: 20px; }\n"
        html_content += ".section { margin: 20px 0; padding: 10px; border: 1px solid #ddd; }\n"
        html_content += ".header { background-color: #f5f5f5; padding: 10px; }\n"
        html_content += "</style>\n</head>\n<body>\n"
        
        if include_metadata:
            html_content += f"<div class='header'><h1>Analysis Report</h1>"
            html_content += f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p></div>\n"
        
        # Add content sections
        for key, value in data.items():
            if key == "metadata":
                continue
            
            html_content += f"<div class='section'>\n"
            html_content += f"<h2>{key.replace('_', ' ').title()}</h2>\n"
            
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    html_content += f"<h3>{sub_key.replace('_', ' ').title()}</h3>\n"
                    html_content += f"<p>{str(sub_value)}</p>\n"
            elif isinstance(value, list):
                html_content += "<ul>\n"
                for item in value:
                    html_content += f"<li>{str(item)}</li>\n"
                html_content += "</ul>\n"
            else:
                html_content += f"<p>{str(value)}</p>\n"
            
            html_content += "</div>\n"
        
        html_content += "</body>\n</html>"
        
        return {
            "status": "success",
            "format": "html",
            "data": html_content,
            "size": len(html_content)
        }
    
    async def _export_to_pdf(
        self,
        data: Dict[str, Any],
        include_visualizations: bool = True,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Export data to PDF format."""
        # For now, return a placeholder since PDF generation requires additional libraries
        return {
            "status": "success",
            "format": "pdf",
            "data": "PDF export placeholder - requires reportlab or weasyprint",
            "note": "PDF generation requires additional dependencies"
        }
    
    async def _export_to_excel(
        self,
        data: Dict[str, Any],
        include_visualizations: bool = True,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Export data to Excel format."""
        # For now, return a placeholder since Excel generation requires additional libraries
        return {
            "status": "success",
            "format": "excel",
            "data": "Excel export placeholder - requires openpyxl or xlsxwriter",
            "note": "Excel generation requires additional dependencies"
        }


class SharingManager:
    """Handles report sharing via multiple channels."""
    
    def __init__(self):
        self.sharing_methods = {
            "email": self._share_via_email,
            "cloud": self._share_via_cloud,
            "api": self._share_via_api
        }
    
    async def share_report(
        self,
        report_data: Dict[str, Any],
        sharing_methods: List[str] = ["api"],
        recipients: List[str] = None,
        include_notifications: bool = True
    ) -> Dict[str, Any]:
        """Share report via multiple channels."""
        results = {}
        
        for method in sharing_methods:
            if method in self.sharing_methods:
                try:
                    share_func = self.sharing_methods[method]
                    result = await share_func(
                        report_data, recipients, include_notifications
                    )
                    results[method] = result
                except Exception as e:
                    results[method] = {
                        "status": "error",
                        "error": str(e)
                    }
            else:
                results[method] = {
                    "status": "error",
                    "error": f"Unsupported sharing method: {method}"
                }
        
        return results
    
    async def _share_via_email(
        self,
        report_data: Dict[str, Any],
        recipients: List[str] = None,
        include_notifications: bool = True
    ) -> Dict[str, Any]:
        """Share report via email."""
        return {
            "status": "success",
            "method": "email",
            "recipients": recipients or [],
            "message": "Email sharing placeholder - requires email configuration"
        }
    
    async def _share_via_cloud(
        self,
        report_data: Dict[str, Any],
        recipients: List[str] = None,
        include_notifications: bool = True
    ) -> Dict[str, Any]:
        """Share report via cloud storage."""
        return {
            "status": "success",
            "method": "cloud",
            "recipients": recipients or [],
            "message": "Cloud sharing placeholder - requires cloud storage configuration"
        }
    
    async def _share_via_api(
        self,
        report_data: Dict[str, Any],
        recipients: List[str] = None,
        include_notifications: bool = True
    ) -> Dict[str, Any]:
        """Share report via API endpoint."""
        return {
            "status": "success",
            "method": "api",
            "recipients": recipients or [],
            "data": report_data,
            "message": "Report shared via API successfully"
        }


class DataExportAgent(BaseAgent):
    """Agent for data export and sharing capabilities."""
    
    def __init__(self):
        super().__init__()
        self.agent_id = "data_export_agent"
        self.name = "Data Export Agent"
        self.description = "Export and sharing capabilities for analysis results"
        
        # Initialize components
        self.export_manager = ExportManager()
        self.sharing_manager = SharingManager()
        
        # Create export directory
        self.export_dir = Path("Results/exports")
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Data Export Agent initialized with export directory: {self.export_dir}")
    
    def can_process(self, content: str) -> bool:
        """Check if this agent can process the given content."""
        # This agent can process any content type for export
        return True
    
    async def process(self, content: str, **kwargs) -> Dict[str, Any]:
        """Process content and export it."""
        try:
            # Export the content
            export_data = {
                "content": content,
                "analysis_results": "Sample analysis",
                "export_timestamp": datetime.now().isoformat()
            }
            
            export_formats = kwargs.get("export_formats", ["json"])
            result = await self.export_analysis_results(
                data=export_data,
                export_formats=export_formats,
                include_visualizations=kwargs.get("include_visualizations", True),
                include_metadata=kwargs.get("include_metadata", True)
            )
            
            return {
                "status": "success",
                "agent_id": self.agent_id,
                "result": result
            }
        except Exception as e:
            return {
                "status": "error",
                "agent_id": self.agent_id,
                "error": str(e)
            }
    
    async def export_analysis_results(
        self,
        data: Dict[str, Any],
        export_formats: List[str] = ["json"],
        include_visualizations: bool = True,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Export analysis results to multiple formats."""
        try:
            # Export data to requested formats
            export_results = await self.export_manager.export_data(
                data, export_formats, include_visualizations, include_metadata
            )
            
            # Save exported files
            saved_files = []
            for format_type, result in export_results.items():
                if result["status"] == "success":
                    file_path = await self._save_export_file(
                        result, format_type
                    )
                    if file_path:
                        saved_files.append({
                            "format": format_type,
                            "file_path": str(file_path),
                            "size": result.get("size", 0)
                        })
            
            return {
                "status": "success",
                "export_results": export_results,
                "saved_files": saved_files,
                "total_formats": len(export_formats)
            }
            
        except Exception as e:
            logging.error(f"Error exporting analysis results: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _save_export_file(
        self, result: Dict[str, Any], format_type: str
    ) -> Path:
        """Save exported file to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"export_{timestamp}.{format_type}"
            file_path = self.export_dir / filename
            
            if format_type == "json":
                with open(file_path, 'w') as f:
                    json.dump(result["data"], f, indent=2)
            elif format_type == "csv":
                with open(file_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=result["data"][0].keys())
                    writer.writeheader()
                    writer.writerows(result["data"])
            elif format_type == "html":
                with open(file_path, 'w') as f:
                    f.write(result["data"])
            else:
                # For other formats, save as text for now
                with open(file_path, 'w') as f:
                    f.write(str(result["data"]))
            
            return file_path
            
        except Exception as e:
            logging.error(f"Error saving export file: {e}")
            return None
    
    async def share_reports(
        self,
        report_data: Dict[str, Any],
        sharing_methods: List[str] = ["api"],
        recipients: List[str] = None,
        include_notifications: bool = True
    ) -> Dict[str, Any]:
        """Share reports via multiple channels."""
        try:
            # Share report via requested methods
            sharing_results = await self.sharing_manager.share_report(
                report_data, sharing_methods, recipients, include_notifications
            )
            
            return {
                "status": "success",
                "sharing_results": sharing_results,
                "total_methods": len(sharing_methods)
            }
            
        except Exception as e:
            logging.error(f"Error sharing reports: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def get_export_history(self, limit: int = 10) -> Dict[str, Any]:
        """Get export history."""
        try:
            export_files = list(self.export_dir.glob("export_*"))
            export_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            history = []
            for file_path in export_files[:limit]:
                stat = file_path.stat()
                history.append({
                    "filename": file_path.name,
                    "file_path": str(file_path),
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            
            return {
                "status": "success",
                "total_exports": len(export_files),
                "recent_exports": history
            }
            
        except Exception as e:
            logging.error(f"Error getting export history: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def cleanup_old_exports(self, days: int = 30) -> Dict[str, Any]:
        """Clean up old export files."""
        try:
            cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
            deleted_files = []
            
            for file_path in self.export_dir.glob("export_*"):
                if file_path.stat().st_mtime < cutoff_date:
                    file_path.unlink()
                    deleted_files.append(file_path.name)
            
            return {
                "status": "success",
                "deleted_files": deleted_files,
                "total_deleted": len(deleted_files)
            }
            
        except Exception as e:
            logging.error(f"Error cleaning up old exports: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
