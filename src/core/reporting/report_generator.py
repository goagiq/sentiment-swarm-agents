"""
Report Generator

Provides automated report generation capabilities including:
- Data analysis reports
- Performance reports
- Trend analysis reports
- Custom report templates
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Automated report generation system."""
    
    def __init__(self, output_dir: str = "reports"):
        """Initialize the report generator.
        
        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Report templates
        self.templates = {
            "executive_summary": self._executive_summary_template,
            "performance_analysis": self._performance_analysis_template,
            "trend_report": self._trend_report_template,
            "custom": self._custom_template
        }
    
    def generate_report(self, 
                       report_type: str,
                       data: Dict[str, Any],
                       template: str = "custom",
                       format: str = "json") -> Dict[str, Any]:
        """Generate a report based on type and data.
        
        Args:
            report_type: Type of report to generate
            data: Data to include in the report
            template: Template to use for formatting
            format: Output format (json, html, pdf)
            
        Returns:
            Report data and metadata
        """
        try:
            # Get template function
            template_func = self.templates.get(template, self._custom_template)
            
            # Generate report content
            report_content = template_func(report_type, data)
            
            # Add metadata
            report_content.update({
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "report_type": report_type,
                    "template": template,
                    "format": format
                }
            })
            
            # Save report
            filename = self._save_report(report_content, report_type, format)
            
            return {
                "success": True,
                "report": report_content,
                "filename": filename,
                "message": f"Report generated successfully: {filename}"
            }
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to generate report"
            }
    
    def _executive_summary_template(self, report_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Executive summary report template."""
        return {
            "title": f"Executive Summary - {report_type.title()}",
            "summary": {
                "key_insights": data.get("key_insights", []),
                "recommendations": data.get("recommendations", []),
                "metrics": data.get("metrics", {}),
                "trends": data.get("trends", [])
            },
            "sections": [
                {
                    "title": "Executive Overview",
                    "content": data.get("overview", "No overview available")
                },
                {
                    "title": "Key Performance Indicators",
                    "content": data.get("kpis", {})
                },
                {
                    "title": "Strategic Recommendations",
                    "content": data.get("strategic_recommendations", [])
                }
            ]
        }
    
    def _performance_analysis_template(self, report_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Performance analysis report template."""
        return {
            "title": f"Performance Analysis - {report_type.title()}",
            "summary": {
                "performance_score": data.get("performance_score", 0),
                "benchmark_comparison": data.get("benchmark_comparison", {}),
                "improvement_areas": data.get("improvement_areas", [])
            },
            "sections": [
                {
                    "title": "Performance Metrics",
                    "content": data.get("performance_metrics", {})
                },
                {
                    "title": "Trend Analysis",
                    "content": data.get("trend_analysis", {})
                },
                {
                    "title": "Bottleneck Analysis",
                    "content": data.get("bottlenecks", [])
                },
                {
                    "title": "Optimization Opportunities",
                    "content": data.get("optimization_opportunities", [])
                }
            ]
        }
    
    def _trend_report_template(self, report_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Trend analysis report template."""
        return {
            "title": f"Trend Analysis - {report_type.title()}",
            "summary": {
                "trend_direction": data.get("trend_direction", "stable"),
                "trend_strength": data.get("trend_strength", 0),
                "seasonality": data.get("seasonality", {}),
                "forecast": data.get("forecast", {})
            },
            "sections": [
                {
                    "title": "Historical Trends",
                    "content": data.get("historical_trends", {})
                },
                {
                    "title": "Seasonal Patterns",
                    "content": data.get("seasonal_patterns", {})
                },
                {
                    "title": "Forecast Analysis",
                    "content": data.get("forecast_analysis", {})
                },
                {
                    "title": "Anomaly Detection",
                    "content": data.get("anomalies", [])
                }
            ]
        }
    
    def _custom_template(self, report_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Custom report template."""
        return {
            "title": f"Custom Report - {report_type.title()}",
            "summary": data.get("summary", {}),
            "sections": data.get("sections", []),
            "raw_data": data.get("raw_data", {})
        }
    
    def _save_report(self, report_content: Dict[str, Any], report_type: str, format: str) -> str:
        """Save report to file.
        
        Args:
            report_content: Report content to save
            report_type: Type of report
            format: File format
            
        Returns:
            Filename of saved report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_type}_{timestamp}.{format}"
        filepath = self.output_dir / filename
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(report_content, f, indent=2)
        elif format == "html":
            html_content = self._convert_to_html(report_content)
            with open(filepath, 'w') as f:
                f.write(html_content)
        else:
            # Default to JSON
            with open(filepath, 'w') as f:
                json.dump(report_content, f, indent=2)
        
        return str(filepath)
    
    def _convert_to_html(self, report_content: Dict[str, Any]) -> str:
        """Convert report content to HTML format."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_content.get('title', 'Report')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007bff; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report_content.get('title', 'Report')}</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # Add summary
        if "summary" in report_content:
            html += "<div class='section'><h2>Summary</h2>"
            for key, value in report_content["summary"].items():
                html += f"<div class='metric'><strong>{key}:</strong> {value}</div>"
            html += "</div>"
        
        # Add sections
        for section in report_content.get("sections", []):
            html += f"""
            <div class='section'>
                <h2>{section.get('title', 'Section')}</h2>
                <p>{section.get('content', 'No content available')}</p>
            </div>
            """
        
        html += "</body></html>"
        return html
    
    def list_reports(self) -> List[Dict[str, Any]]:
        """List all generated reports.
        
        Returns:
            List of report metadata
        """
        reports = []
        for filepath in self.output_dir.glob("*.*"):
            if filepath.suffix in ['.json', '.html', '.pdf']:
                reports.append({
                    "filename": filepath.name,
                    "path": str(filepath),
                    "size": filepath.stat().st_size,
                    "created": datetime.fromtimestamp(filepath.stat().st_ctime).isoformat()
                })
        return reports
    
    def get_report(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get a specific report by filename.
        
        Args:
            filename: Name of the report file
            
        Returns:
            Report content or None if not found
        """
        filepath = self.output_dir / filename
        if not filepath.exists():
            return None
        
        try:
            if filepath.suffix == '.json':
                with open(filepath, 'r') as f:
                    return json.load(f)
            else:
                with open(filepath, 'r') as f:
                    return {"content": f.read(), "format": filepath.suffix[1:]}
        except Exception as e:
            logger.error(f"Error reading report {filename}: {e}")
            return None
