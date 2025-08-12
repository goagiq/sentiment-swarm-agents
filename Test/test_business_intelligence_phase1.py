#!/usr/bin/env python3
"""
Test script for Phase 1 Business Intelligence functionality.
Tests business dashboards, executive reporting, and data visualizations.
"""

import asyncio
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.business_intelligence_agent import BusinessIntelligenceAgent
from agents.data_visualization_agent import DataVisualizationAgent
from core.models import AnalysisRequest, DataType


class BusinessIntelligencePhase1Tester:
    """Test Phase 1 Business Intelligence functionality."""
    
    def __init__(self):
        self.bi_agent = BusinessIntelligenceAgent()
        self.viz_agent = DataVisualizationAgent()
        self.test_results = []
        
    async def test_business_dashboard_generation(self):
        """Test business dashboard generation."""
        print("🔧 Testing Business Dashboard Generation...")
        
        try:
            # Test data
            test_data = "Sample business data for dashboard testing. This includes various metrics and KPIs."
            
            # Test different dashboard types
            dashboard_types = ["executive", "detailed", "comprehensive"]
            
            for dashboard_type in dashboard_types:
                print(f"  - Testing {dashboard_type} dashboard...")
                
                result = await self.bi_agent.generate_business_dashboard(
                    test_data, 
                    dashboard_type
                )
                
                if result and not result.get("error"):
                    print(f"    ✅ {dashboard_type} dashboard generated successfully")
                    self.test_results.append({
                        "test": f"dashboard_{dashboard_type}",
                        "status": "passed",
                        "result": result
                    })
                else:
                    print(f"    ❌ {dashboard_type} dashboard generation failed")
                    self.test_results.append({
                        "test": f"dashboard_{dashboard_type}",
                        "status": "failed",
                        "error": result.get("error", "Unknown error")
                    })
            
            return True
            
        except Exception as e:
            print(f"    ❌ Dashboard generation test failed: {e}")
            self.test_results.append({
                "test": "dashboard_generation",
                "status": "failed",
                "error": str(e)
            })
            return False
    
    async def test_executive_reporting(self):
        """Test executive reporting functionality."""
        print("🔧 Testing Executive Reporting...")
        
        try:
            # Test data
            test_content = """
            Quarterly Business Report Summary:
            
            Our company has shown strong performance in Q3 2024 with revenue growth of 15% 
            compared to the previous quarter. Customer satisfaction scores have improved 
            to 4.2/5.0, and we've successfully launched three new product features.
            
            Key achievements:
            - Revenue growth: 15% quarter-over-quarter
            - Customer satisfaction: 4.2/5.0
            - New product features launched: 3
            - Market share increase: 2.5%
            
            Challenges and opportunities:
            - Supply chain optimization needed
            - Expansion into new markets planned
            - Technology infrastructure upgrades required
            """
            
            # Test executive report generation
            print("  - Testing executive report generation...")
            
            result = await self.bi_agent.generate_executive_report(
                test_content,
                "comprehensive"
            )
            
            if result and not result.get("error"):
                print("    ✅ Executive report generated successfully")
                self.test_results.append({
                    "test": "executive_report",
                    "status": "passed",
                    "result": result
                })
            else:
                print("    ❌ Executive report generation failed")
                self.test_results.append({
                    "test": "executive_report",
                    "status": "failed",
                    "error": result.get("error", "Unknown error")
                })
            
            return True
            
        except Exception as e:
            print(f"    ❌ Executive reporting test failed: {e}")
            self.test_results.append({
                "test": "executive_reporting",
                "status": "failed",
                "error": str(e)
            })
            return False
    
    async def test_data_visualizations(self):
        """Test data visualization functionality."""
        print("🔧 Testing Data Visualizations...")
        
        try:
            # Test data
            test_data = "Sample data for visualization testing with various metrics and trends."
            
            # Test different chart types
            chart_types = ["trend", "distribution", "correlation", "pie", "bar", "scatter"]
            
            for chart_type in chart_types:
                print(f"  - Testing {chart_type} chart...")
                
                result = await self.viz_agent.generate_visualizations(
                    test_data,
                    [chart_type],
                    True
                )
                
                if result and not result.get("error"):
                    print(f"    ✅ {chart_type} chart generated successfully")
                    self.test_results.append({
                        "test": f"visualization_{chart_type}",
                        "status": "passed",
                        "result": result
                    })
                else:
                    print(f"    ❌ {chart_type} chart generation failed")
                    self.test_results.append({
                        "test": f"visualization_{chart_type}",
                        "status": "failed",
                        "error": result.get("error", "Unknown error")
                    })
            
            return True
            
        except Exception as e:
            print(f"    ❌ Data visualization test failed: {e}")
            self.test_results.append({
                "test": "data_visualization",
                "status": "failed",
                "error": str(e)
            })
            return False
    
    async def test_trend_analysis(self):
        """Test trend analysis functionality."""
        print("🔧 Testing Trend Analysis...")
        
        try:
            # Test data
            test_data = """
            Historical performance data:
            January: 100 units sold, $10,000 revenue
            February: 120 units sold, $12,000 revenue
            March: 115 units sold, $11,500 revenue
            April: 140 units sold, $14,000 revenue
            May: 150 units sold, $15,000 revenue
            """
            
            # Test trend analysis
            print("  - Testing business trends analysis...")
            
            result = await self.bi_agent.analyze_business_trends(
                test_data,
                "30d"
            )
            
            if result and not result.get("error"):
                print("    ✅ Trend analysis completed successfully")
                self.test_results.append({
                    "test": "trend_analysis",
                    "status": "passed",
                    "result": result
                })
            else:
                print("    ❌ Trend analysis failed")
                self.test_results.append({
                    "test": "trend_analysis",
                    "status": "failed",
                    "error": result.get("error", "Unknown error")
                })
            
            return True
            
        except Exception as e:
            print(f"    ❌ Trend analysis test failed: {e}")
            self.test_results.append({
                "test": "trend_analysis",
                "status": "failed",
                "error": str(e)
            })
            return False
    
    async def test_comprehensive_analysis(self):
        """Test comprehensive business intelligence analysis."""
        print("🔧 Testing Comprehensive Analysis...")
        
        try:
            # Test data
            test_content = """
            Comprehensive Business Analysis Data:
            
            Financial Performance:
            - Revenue: $1.2M (Q3 2024)
            - Growth Rate: 15% YoY
            - Profit Margin: 25%
            
            Customer Metrics:
            - Customer Satisfaction: 4.3/5.0
            - Net Promoter Score: 65
            - Customer Retention: 92%
            
            Operational Metrics:
            - Employee Satisfaction: 4.1/5.0
            - Process Efficiency: 87%
            - Quality Score: 94%
            
            Market Position:
            - Market Share: 12%
            - Competitive Position: Strong
            - Brand Recognition: 78%
            """
            
            # Create analysis request
            request = AnalysisRequest(
                data_type=DataType.TEXT,
                content=test_content,
                language="en",
                metadata={
                    "request_type": "comprehensive",
                    "include_visualizations": True,
                    "include_insights": True
                }
            )
            
            # Test comprehensive analysis
            print("  - Testing comprehensive business analysis...")
            
            result = await self.bi_agent.process(request)
            
            if result and result.status == "completed":
                print("    ✅ Comprehensive analysis completed successfully")
                self.test_results.append({
                    "test": "comprehensive_analysis",
                    "status": "passed",
                    "result": {
                        "status": result.status,
                        "processing_time": result.processing_time,
                        "metadata": result.metadata
                    }
                })
            else:
                print("    ❌ Comprehensive analysis failed")
                self.test_results.append({
                    "test": "comprehensive_analysis",
                    "status": "failed",
                    "error": "Analysis did not complete successfully"
                })
            
            return True
            
        except Exception as e:
            print(f"    ❌ Comprehensive analysis test failed: {e}")
            self.test_results.append({
                "test": "comprehensive_analysis",
                "status": "failed",
                "error": str(e)
            })
            return False
    
    async def run_all_tests(self):
        """Run all Phase 1 business intelligence tests."""
        print("🚀 Starting Phase 1 Business Intelligence Tests")
        print("=" * 60)
        
        test_functions = [
            self.test_business_dashboard_generation,
            self.test_executive_reporting,
            self.test_data_visualizations,
            self.test_trend_analysis,
            self.test_comprehensive_analysis
        ]
        
        passed_tests = 0
        total_tests = len(test_functions)
        
        for test_func in test_functions:
            try:
                result = await test_func()
                if result:
                    passed_tests += 1
            except Exception as e:
                print(f"❌ Test {test_func.__name__} failed with exception: {e}")
        
        # Generate test report
        await self.generate_test_report(passed_tests, total_tests)
        
        print("=" * 60)
        print(f"✅ Phase 1 Business Intelligence Tests Completed")
        print(f"   Passed: {passed_tests}/{total_tests}")
        print(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        return passed_tests == total_tests
    
    async def generate_test_report(self, passed_tests: int, total_tests: int):
        """Generate a test report."""
        try:
            # Create Results directory if it doesn't exist
            results_dir = Path("../Results")
            results_dir.mkdir(exist_ok=True)
            
            # Create business intelligence results directory
            bi_results_dir = results_dir / "business_intelligence"
            bi_results_dir.mkdir(exist_ok=True)
            
            # Generate report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = bi_results_dir / f"phase1_test_results_{timestamp}.json"
            
            report = {
                "test_suite": "Phase 1 Business Intelligence",
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": total_tests - passed_tests,
                    "success_rate": (passed_tests/total_tests)*100
                },
                "test_results": self.test_results
            }
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if hasattr(obj, 'tolist'):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            report = convert_numpy(report)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"📊 Test report saved to: {report_path}")
            
        except Exception as e:
            print(f"❌ Failed to generate test report: {e}")


async def main():
    """Main test function."""
    tester = BusinessIntelligencePhase1Tester()
    success = await tester.run_all_tests()
    
    if success:
        print("🎉 All Phase 1 Business Intelligence tests passed!")
        return 0
    else:
        print("⚠️ Some Phase 1 Business Intelligence tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
