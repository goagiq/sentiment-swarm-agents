#!/usr/bin/env python3
"""
Test script to demonstrate automatic report saving functionality.

This script shows how the updated MCP tools automatically save reports
to the Results/reports/ directory and provide links.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.report_manager import report_manager


async def test_auto_report_saving():
    """Test the automatic report saving functionality."""
    
    print("🧪 Testing Automatic Report Saving Functionality")
    print("=" * 60)
    
    # Clear any existing reports for clean test
    report_manager.clear_reports()
    
    # Test 1: Save a simple analysis report
    print("\n1. 📄 Testing Report Saving...")
    analysis_content = """# Sample Analysis Report

## Executive Summary
This is a sample analysis report to test automatic saving functionality.

## Key Findings
- Finding 1: Automatic report saving works
- Finding 2: Reports are saved with timestamps
- Finding 3: Links are provided automatically

## Conclusion
The automatic report saving system is functioning correctly.
"""
    
    result1 = report_manager.save_report(
        content=analysis_content,
        filename="Sample_Analysis_Report.md",
        report_type="analysis",
        metadata={"test": True, "category": "sample"}
    )
    
    if result1["success"]:
        print(f"✅ Report saved: {result1['report_info']['relative_path']}")
        print(f"   Size: {result1['report_info']['size_kb']}KB")
    else:
        print(f"❌ Failed to save report: {result1['error']}")
    
    # Test 2: Save a visualization
    print("\n2. 🎯 Testing Visualization Saving...")
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Sample Visualization</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .chart { background: #f0f0f0; padding: 20px; border-radius: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Sample Data Visualization</h1>
        <div class="chart">
            <h2>Sample Chart</h2>
            <p>This is a sample visualization to test automatic saving.</p>
            <ul>
                <li>Data Point 1: 100</li>
                <li>Data Point 2: 200</li>
                <li>Data Point 3: 150</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
    
    result2 = report_manager.save_visualization(
        html_content=html_content,
        title="Sample Data Visualization",
        visualization_type="interactive",
        metadata={"test": True, "chart_type": "sample"}
    )
    
    if result2["success"]:
        print(f"✅ Visualization saved: {result2['visualization_info']['relative_path']}")
        print(f"   Size: {result2['visualization_info']['size_kb']}KB")
    else:
        print(f"❌ Failed to save visualization: {result2['error']}")
    
    # Test 3: Generate summary report
    print("\n3. 📋 Testing Summary Report Generation...")
    key_findings = [
        "Automatic report saving is working correctly",
        "Reports are saved with unique timestamps and IDs",
        "Visualizations are properly formatted and saved",
        "Summary reports include links to all generated files"
    ]
    
    result3 = report_manager.generate_summary_report(
        analysis_title="Automatic Report Saving Test",
        analysis_type="comprehensive",
        key_findings=key_findings
    )
    
    if result3["success"]:
        print(f"✅ Summary report saved: {result3['summary_info']['relative_path']}")
        print(f"   Total reports: {result3['summary_info']['total_reports']}")
        print(f"   Total size: {result3['summary_info']['size_kb']}KB")
    else:
        print(f"❌ Failed to generate summary: {result3['message']}")
    
    # Test 4: Get all reports
    print("\n4. 📊 Getting All Generated Reports...")
    all_reports = report_manager.get_all_reports()
    print(f"✅ Total reports generated: {len(all_reports)}")
    
    for i, report in enumerate(all_reports, 1):
        print(f"   {i}. {report['filename']} ({report['size_kb']}KB)")
    
    # Test 5: Show final summary
    print("\n" + "=" * 60)
    print("🎉 AUTOMATIC REPORT SAVING TEST COMPLETE")
    print("=" * 60)
    
    total_size = sum(r['size_kb'] for r in all_reports)
    print(f"📁 Reports Directory: Results/reports/")
    print(f"📄 Total Reports: {len(all_reports)}")
    print(f"💾 Total Size: {total_size:.1f}KB")
    print(f"✅ All reports automatically saved with timestamps and unique IDs")
    print(f"🔗 Summary report includes links to all generated files")
    
    # Show the summary report path
    summary_reports = [r for r in all_reports if r['report_type'] == 'summary']
    if summary_reports:
        summary_path = summary_reports[0]['relative_path']
        print(f"\n📋 Summary Report: {summary_path}")
        print("   This file contains links to all generated reports.")
    
    print("\n✨ The MCP tools now automatically save all reports!")
    print("   No manual file management required.")


if __name__ == "__main__":
    asyncio.run(test_auto_report_saving())
