"""
Report Manager Service for automatic report saving and link generation.

This service automatically saves all generated reports to the Results/reports/
directory and provides standardized links and summaries.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from uuid import uuid4

from loguru import logger


class ReportManager:
    """Manages automatic report saving and link generation."""
    
    def __init__(self):
        """Initialize the report manager."""
        self.reports_dir = Path("Results/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.generated_reports: List[Dict[str, Any]] = []
        
        logger.info(f"Report Manager initialized. Reports directory: {self.reports_dir.absolute()}")
    
    def save_report(
        self,
        content: str,
        filename: str,
        report_type: str = "analysis",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Save a report to the reports directory."""
        try:
            # Generate timestamp for unique filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid4())[:8]
            
            # Create filename with timestamp and unique ID
            base_name = Path(filename).stem
            extension = Path(filename).suffix
            final_filename = f"{base_name}_{timestamp}_{unique_id}{extension}"
            
            # Full path for the report
            report_path = self.reports_dir / final_filename
            
            # Save the report
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Create report metadata
            report_info = {
                "filename": final_filename,
                "path": str(report_path),
                "relative_path": f"Results/reports/{final_filename}",
                "report_type": report_type,
                "size_bytes": len(content),
                "size_kb": round(len(content) / 1024, 2),
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            # Add to generated reports list
            self.generated_reports.append(report_info)
            
            logger.info(f"Report saved: {report_path} ({report_info['size_kb']}KB)")
            
            return {
                "success": True,
                "report_info": report_info,
                "message": f"Report saved successfully to {report_path}"
            }
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to save report"
            }
    
    def save_visualization(
        self,
        html_content: str,
        title: str,
        visualization_type: str = "interactive",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Save an HTML visualization to the reports directory."""
        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid4())[:8]
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title.replace(' ', '_')
            
            filename = f"{safe_title}_Visualization_{timestamp}_{unique_id}.html"
            report_path = self.reports_dir / filename
            
            # Save the HTML visualization
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Create visualization metadata
            viz_info = {
                "filename": filename,
                "path": str(report_path),
                "relative_path": f"Results/reports/{filename}",
                "report_type": "visualization",
                "visualization_type": visualization_type,
                "title": title,
                "size_bytes": len(html_content),
                "size_kb": round(len(html_content) / 1024, 2),
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            # Add to generated reports list
            self.generated_reports.append(viz_info)
            
            logger.info(f"Visualization saved: {report_path} ({viz_info['size_kb']}KB)")
            
            return {
                "success": True,
                "visualization_info": viz_info,
                "message": f"Visualization saved successfully to {report_path}"
            }
            
        except Exception as e:
            logger.error(f"Error saving visualization: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to save visualization"
            }
    
    def generate_summary_report(
        self,
        analysis_title: str,
        analysis_type: str = "comprehensive",
        key_findings: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate a summary report with links to all generated reports."""
        try:
            if not self.generated_reports:
                return {
                    "success": False,
                    "message": "No reports to summarize"
                }
            
            # Generate summary content
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid4())[:8]
            
            summary_content = self._create_summary_content(
                analysis_title, analysis_type, key_findings
            )
            
            filename = f"{analysis_title.replace(' ', '_')}_Summary_{timestamp}_{unique_id}.md"
            summary_path = self.reports_dir / filename
            
            # Save summary report
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_content)
            
            # Create summary metadata
            summary_info = {
                "filename": filename,
                "path": str(summary_path),
                "relative_path": f"Results/reports/{filename}",
                "report_type": "summary",
                "analysis_title": analysis_title,
                "analysis_type": analysis_type,
                "total_reports": len(self.generated_reports),
                "size_bytes": len(summary_content),
                "size_kb": round(len(summary_content) / 1024, 2),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add to generated reports list
            self.generated_reports.append(summary_info)
            
            logger.info(f"Summary report saved: {summary_path} ({summary_info['size_kb']}KB)")
            
            return {
                "success": True,
                "summary_info": summary_info,
                "all_reports": self.generated_reports,
                "message": f"Summary report saved successfully to {summary_path}"
            }
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to generate summary report"
            }
    
    def _create_summary_content(
        self,
        analysis_title: str,
        analysis_type: str,
        key_findings: Optional[List[str]] = None
    ) -> str:
        """Create the content for the summary report."""
        content = f"""# {analysis_title}: Complete Report Summary

## Analysis Overview

This comprehensive analysis was automatically generated and all reports have been saved to the `/Results/reports/` directory.

## Generated Reports

"""
        
        # Add report links
        for i, report in enumerate(self.generated_reports, 1):
            report_type_icon = self._get_report_icon(report.get("report_type", "analysis"))
            content += f"""### {i}. {report_type_icon} **{report.get('title', report['filename'])}**
- **File:** `{report['filename']}`
- **Path:** `{report['relative_path']}`
- **Type:** {report.get('report_type', 'analysis').title()}
- **Size:** {report['size_kb']}KB
- **Generated:** {report['timestamp']}

"""
        
        # Add key findings if provided
        if key_findings:
            content += """## Key Findings

"""
            for finding in key_findings:
                content += f"- {finding}\n"
            content += "\n"
        
        # Add quick access links
        content += """## Quick Access Links

### 📁 **Report Directory:** `Results/reports/`

"""
        
        for report in self.generated_reports:
            report_type_icon = self._get_report_icon(report.get("report_type", "analysis"))
            content += f"""### {report_type_icon} **{report.get('title', report['filename'])}:**
`{report['relative_path']}`

"""
        
        # Add footer
        content += f"""---

**Analysis Status:** ✅ Complete  
**Reports Generated:** {len(self.generated_reports)} files  
**Total Size:** {sum(r['size_kb'] for r in self.generated_reports):.1f}KB  
**Location:** `Results/reports/`  
**Analysis Type:** {analysis_type.title()}  
**Scope:** {analysis_title}  
**Generated:** {datetime.now().isoformat()}
"""
        
        return content
    
    def _get_report_icon(self, report_type: str) -> str:
        """Get appropriate icon for report type."""
        icons = {
            "analysis": "📄",
            "visualization": "🎯",
            "summary": "📋",
            "report": "📊",
            "dashboard": "📈"
        }
        return icons.get(report_type, "📄")
    
    def get_all_reports(self) -> List[Dict[str, Any]]:
        """Get all generated reports."""
        return self.generated_reports.copy()
    
    def clear_reports(self):
        """Clear the generated reports list (for new analysis sessions)."""
        self.generated_reports.clear()
        logger.info("Report list cleared for new analysis session")


# Global report manager instance
report_manager = ReportManager()
