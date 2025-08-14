"""
Test script for External System Integration (Phase 8)

This script tests the integration with external business systems including:
- ERP System Connector (SAP, Oracle, Dynamics, NetSuite, Infor)
- CRM System Connector (Salesforce, HubSpot, Dynamics CRM, Pipedrive, Zoho)
- BI Tool Connector (Tableau, Power BI, Qlik, Looker, Sisense)
- Integration Manager (unified interface)

Tests cover:
- Authentication with different systems
- Data retrieval from each system
- Cross-system analytics
- Error handling and fallback mechanisms
- Performance and caching
"""

import asyncio
import sys
import os
import logging
from datetime import datetime
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.integration.erp_connector import (
    ERPConnector, ERPConfig, ERPType, FinancialData, InventoryData, SupplyChainData
)
from core.integration.crm_connector import (
    CRMConnector, CRMConfig, CRMType, CustomerProfile, SalesPipeline, InteractionData
)
from core.integration.bi_connector import (
    BIConnector, BIConfig, BIType, DashboardData, KPIMetric, ReportData
)
from core.integration.integration_manager import (
    IntegrationManager, IntegrationConfig, UnifiedBusinessData
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExternalSystemIntegrationTester:
    """Test suite for external system integration components"""
    
    def __init__(self):
        self.test_results = []
        self.start_time = datetime.now()
        
    def log_test_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        result = {
            "test_name": test_name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"   Details: {details}")
        print()
        
    async def test_erp_connector(self):
        """Test ERP system connector functionality"""
        print("Testing ERP System Connector...")
        
        # Test configuration
        config = ERPConfig(
            erp_type=ERPType.MOCK,
            base_url="https://mock-erp.example.com",
            api_key="test_key"
        )
        
        try:
            async with ERPConnector(config) as erp:
                # Test financial data
                financial_data = await erp.get_financial_data()
                if financial_data and isinstance(financial_data, FinancialData):
                    self.log_test_result(
                        "ERP Financial Data Retrieval",
                        True,
                        f"Revenue: ${financial_data.revenue:,.2f}, Profit Margin: {financial_data.profit_margin:.1%}"
                    )
                else:
                    self.log_test_result("ERP Financial Data Retrieval", False, "No data returned")
                    
                # Test inventory data
                inventory_data = await erp.get_inventory_data()
                if inventory_data and isinstance(inventory_data, InventoryData):
                    self.log_test_result(
                        "ERP Inventory Data Retrieval",
                        True,
                        f"Total Items: {inventory_data.total_items:,}, Value: ${inventory_data.inventory_value:,.2f}"
                    )
                else:
                    self.log_test_result("ERP Inventory Data Retrieval", False, "No data returned")
                    
                # Test supply chain data
                supply_chain_data = await erp.get_supply_chain_data()
                if supply_chain_data and isinstance(supply_chain_data, SupplyChainData):
                    self.log_test_result(
                        "ERP Supply Chain Data Retrieval",
                        True,
                        f"Active Orders: {supply_chain_data.active_orders}, Pending Deliveries: {supply_chain_data.pending_deliveries}"
                    )
                else:
                    self.log_test_result("ERP Supply Chain Data Retrieval", False, "No data returned")
                    
                # Test customer data
                customer_data = await erp.get_customer_data("CUST001")
                if customer_data and "customer_id" in customer_data:
                    self.log_test_result(
                        "ERP Customer Data Retrieval",
                        True,
                        f"Customer: {customer_data.get('name', 'Unknown')}"
                    )
                else:
                    self.log_test_result("ERP Customer Data Retrieval", False, "No data returned")
                    
                # Test production metrics
                production_metrics = await erp.get_production_metrics()
                if production_metrics and "efficiency" in production_metrics:
                    self.log_test_result(
                        "ERP Production Metrics Retrieval",
                        True,
                        f"Efficiency: {production_metrics['efficiency']:.1%}"
                    )
                else:
                    self.log_test_result("ERP Production Metrics Retrieval", False, "No data returned")
                    
        except Exception as e:
            self.log_test_result("ERP Connector Initialization", False, str(e))
            
    async def test_crm_connector(self):
        """Test CRM system connector functionality"""
        print("Testing CRM System Connector...")
        
        # Test configuration
        config = CRMConfig(
            crm_type=CRMType.MOCK,
            base_url="https://mock-crm.example.com",
            api_key="test_key"
        )
        
        try:
            async with CRMConnector(config) as crm:
                # Test customer profile
                customer_profile = await crm.get_customer_profile("CUST001")
                if customer_profile and isinstance(customer_profile, CustomerProfile):
                    self.log_test_result(
                        "CRM Customer Profile Retrieval",
                        True,
                        f"Customer: {customer_profile.name}, Company: {customer_profile.company}"
                    )
                else:
                    self.log_test_result("CRM Customer Profile Retrieval", False, "No data returned")
                    
                # Test sales pipeline
                sales_pipeline = await crm.get_sales_pipeline()
                if sales_pipeline and isinstance(sales_pipeline, SalesPipeline):
                    self.log_test_result(
                        "CRM Sales Pipeline Retrieval",
                        True,
                        f"Total Leads: {sales_pipeline.total_leads}, Conversion Rate: {sales_pipeline.conversion_rate:.1%}"
                    )
                else:
                    self.log_test_result("CRM Sales Pipeline Retrieval", False, "No data returned")
                    
                # Test customer interactions
                interactions = await crm.get_customer_interactions("CUST001", days=30)
                if interactions and isinstance(interactions, list):
                    self.log_test_result(
                        "CRM Customer Interactions Retrieval",
                        True,
                        f"Found {len(interactions)} interactions"
                    )
                else:
                    self.log_test_result("CRM Customer Interactions Retrieval", False, "No data returned")
                    
                # Test lead data
                lead_data = await crm.get_lead_data("LEAD001")
                if lead_data and "lead_id" in lead_data:
                    self.log_test_result(
                        "CRM Lead Data Retrieval",
                        True,
                        f"Lead: {lead_data.get('name', 'Unknown')}, Status: {lead_data.get('status', 'Unknown')}"
                    )
                else:
                    self.log_test_result("CRM Lead Data Retrieval", False, "No data returned")
                    
                # Test sales performance
                sales_performance = await crm.get_sales_performance()
                if sales_performance and "total_revenue" in sales_performance:
                    self.log_test_result(
                        "CRM Sales Performance Retrieval",
                        True,
                        f"Total Revenue: ${sales_performance['total_revenue']:,.2f}"
                    )
                else:
                    self.log_test_result("CRM Sales Performance Retrieval", False, "No data returned")
                    
        except Exception as e:
            self.log_test_result("CRM Connector Initialization", False, str(e))
            
    async def test_bi_connector(self):
        """Test BI tool connector functionality"""
        print("Testing BI Tool Connector...")
        
        # Test configuration
        config = BIConfig(
            bi_type=BIType.MOCK,
            base_url="https://mock-bi.example.com",
            api_key="test_key"
        )
        
        try:
            async with BIConnector(config) as bi:
                # Test dashboard data
                dashboard_data = await bi.get_dashboard_data("DASH001")
                if dashboard_data and isinstance(dashboard_data, DashboardData):
                    self.log_test_result(
                        "BI Dashboard Data Retrieval",
                        True,
                        f"Dashboard: {dashboard_data.name}, Visualizations: {len(dashboard_data.visualizations)}"
                    )
                else:
                    self.log_test_result("BI Dashboard Data Retrieval", False, "No data returned")
                    
                # Test KPI metrics
                kpi_metrics = await bi.get_kpi_metrics()
                if kpi_metrics and isinstance(kpi_metrics, list):
                    self.log_test_result(
                        "BI KPI Metrics Retrieval",
                        True,
                        f"Found {len(kpi_metrics)} KPI metrics"
                    )
                else:
                    self.log_test_result("BI KPI Metrics Retrieval", False, "No data returned")
                    
                # Test report data
                report_data = await bi.get_report_data("REPORT001")
                if report_data and isinstance(report_data, ReportData):
                    self.log_test_result(
                        "BI Report Data Retrieval",
                        True,
                        f"Report: {report_data.name}, Type: {report_data.type}"
                    )
                else:
                    self.log_test_result("BI Report Data Retrieval", False, "No data returned")
                    
                # Test custom query execution
                query_result = await bi.execute_query("SELECT * FROM sales LIMIT 5")
                if query_result and "result" in query_result:
                    self.log_test_result(
                        "BI Custom Query Execution",
                        True,
                        f"Query executed successfully, {query_result['result']['total_rows']} rows returned"
                    )
                else:
                    self.log_test_result("BI Custom Query Execution", False, "No data returned")
                    
                # Test data warehouse info
                warehouse_info = await bi.get_data_warehouse_info()
                if warehouse_info and "databases" in warehouse_info:
                    self.log_test_result(
                        "BI Data Warehouse Info Retrieval",
                        True,
                        f"Found {len(warehouse_info['databases'])} databases"
                    )
                else:
                    self.log_test_result("BI Data Warehouse Info Retrieval", False, "No data returned")
                    
        except Exception as e:
            self.log_test_result("BI Connector Initialization", False, str(e))
            
    async def test_integration_manager(self):
        """Test integration manager functionality"""
        print("Testing Integration Manager...")
        
        # Test configuration
        config = IntegrationConfig(
            erp_config=ERPConfig(erp_type=ERPType.MOCK, base_url="https://mock-erp.example.com"),
            crm_config=CRMConfig(crm_type=CRMType.MOCK, base_url="https://mock-crm.example.com"),
            bi_config=BIConfig(bi_type=BIType.MOCK, base_url="https://mock-bi.example.com"),
            cache_duration=300,
            max_concurrent_requests=5
        )
        
        try:
            async with IntegrationManager(config) as manager:
                # Test unified business data
                unified_data = await manager.get_unified_business_data(["CUST001"])
                if unified_data and isinstance(unified_data, UnifiedBusinessData):
                    self.log_test_result(
                        "Integration Manager Unified Data Retrieval",
                        True,
                        f"Retrieved data from {sum([1 for x in [unified_data.financial_data, unified_data.inventory_data, unified_data.sales_pipeline] if x])} systems"
                    )
                else:
                    self.log_test_result("Integration Manager Unified Data Retrieval", False, "No data returned")
                    
                # Test customer 360 view
                customer_360 = await manager.get_customer_360_view("CUST001")
                if customer_360 and "customer_id" in customer_360:
                    self.log_test_result(
                        "Integration Manager Customer 360 View",
                        True,
                        f"360 view created for customer {customer_360['customer_id']}"
                    )
                else:
                    self.log_test_result("Integration Manager Customer 360 View", False, "No data returned")
                    
                # Test business health score
                health_score = await manager.get_business_health_score()
                if health_score and "overall_score" in health_score:
                    self.log_test_result(
                        "Integration Manager Business Health Score",
                        True,
                        f"Overall Health Score: {health_score['overall_score']:.1f}/100"
                    )
                else:
                    self.log_test_result("Integration Manager Business Health Score", False, "No data returned")
                    
                # Test individual system data retrieval
                financial_data = await manager.get_financial_data()
                if financial_data:
                    self.log_test_result(
                        "Integration Manager ERP Financial Data",
                        True,
                        f"Revenue: ${financial_data.revenue:,.2f}"
                    )
                else:
                    self.log_test_result("Integration Manager ERP Financial Data", False, "No data returned")
                    
                sales_pipeline = await manager.get_sales_pipeline()
                if sales_pipeline:
                    self.log_test_result(
                        "Integration Manager CRM Sales Pipeline",
                        True,
                        f"Total Leads: {sales_pipeline.total_leads}"
                    )
                else:
                    self.log_test_result("Integration Manager CRM Sales Pipeline", False, "No data returned")
                    
                kpi_metrics = await manager.get_kpi_metrics()
                if kpi_metrics:
                    self.log_test_result(
                        "Integration Manager BI KPI Metrics",
                        True,
                        f"Retrieved {len(kpi_metrics)} KPI metrics"
                    )
                else:
                    self.log_test_result("Integration Manager BI KPI Metrics", False, "No data returned")
                    
                # Test caching functionality
                await manager.clear_cache()
                system_status = await manager.get_system_status()
                if system_status and "cache_size" in system_status:
                    self.log_test_result(
                        "Integration Manager Cache Management",
                        True,
                        f"Cache cleared, size: {system_status['cache_size']}"
                    )
                else:
                    self.log_test_result("Integration Manager Cache Management", False, "Cache operation failed")
                    
        except Exception as e:
            self.log_test_result("Integration Manager Initialization", False, str(e))
            
    async def test_cross_system_analytics(self):
        """Test cross-system analytics capabilities"""
        print("Testing Cross-System Analytics...")
        
        config = IntegrationConfig(
            erp_config=ERPConfig(erp_type=ERPType.MOCK, base_url="https://mock-erp.example.com"),
            crm_config=CRMConfig(crm_type=CRMType.MOCK, base_url="https://mock-crm.example.com"),
            bi_config=BIConfig(bi_type=BIType.MOCK, base_url="https://mock-bi.example.com")
        )
        
        try:
            async with IntegrationManager(config) as manager:
                # Test comprehensive business analysis
                unified_data = await manager.get_unified_business_data(["CUST001", "CUST002"])
                
                if unified_data:
                    # Analyze data completeness
                    data_sources = []
                    if unified_data.financial_data:
                        data_sources.append("ERP Financial")
                    if unified_data.inventory_data:
                        data_sources.append("ERP Inventory")
                    if unified_data.sales_pipeline:
                        data_sources.append("CRM Sales")
                    if unified_data.kpi_metrics:
                        data_sources.append("BI KPIs")
                        
                    self.log_test_result(
                        "Cross-System Data Integration",
                        len(data_sources) >= 3,
                        f"Integrated data from {len(data_sources)} systems: {', '.join(data_sources)}"
                    )
                    
                    # Test business insights generation
                    if unified_data.financial_data and unified_data.sales_pipeline:
                        revenue = unified_data.financial_data.revenue
                        leads = unified_data.sales_pipeline.total_leads
                        avg_deal_size = revenue / leads if leads > 0 else 0
                        
                        self.log_test_result(
                            "Cross-System Business Insights",
                            True,
                            f"Average Deal Size: ${avg_deal_size:,.2f} from {leads} leads"
                        )
                    else:
                        self.log_test_result("Cross-System Business Insights", False, "Insufficient data")
                        
                else:
                    self.log_test_result("Cross-System Data Integration", False, "No unified data retrieved")
                    
        except Exception as e:
            self.log_test_result("Cross-System Analytics", False, str(e))
            
    async def test_error_handling_and_fallbacks(self):
        """Test error handling and fallback mechanisms"""
        print("Testing Error Handling and Fallbacks...")
        
        # Test with invalid configurations
        invalid_config = ERPConfig(
            erp_type=ERPType.SAP,
            base_url="https://invalid-url.example.com",
            api_key="invalid_key"
        )
        
        try:
            async with ERPConnector(invalid_config) as erp:
                # Should fall back to mock data
                financial_data = await erp.get_financial_data()
                if financial_data:
                    self.log_test_result(
                        "Error Handling - Fallback to Mock Data",
                        True,
                        "Successfully fell back to mock data when real system unavailable"
                    )
                else:
                    self.log_test_result("Error Handling - Fallback to Mock Data", False, "No fallback data")
                    
        except Exception as e:
            self.log_test_result("Error Handling - System Initialization", False, str(e))
            
    async def test_performance_and_caching(self):
        """Test performance and caching mechanisms"""
        print("Testing Performance and Caching...")
        
        config = IntegrationConfig(
            erp_config=ERPConfig(erp_type=ERPType.MOCK, base_url="https://mock-erp.example.com"),
            cache_duration=60  # 1 minute cache
        )
        
        try:
            async with IntegrationManager(config) as manager:
                # First call - should cache
                start_time = datetime.now()
                financial_data1 = await manager.get_financial_data()
                first_call_time = (datetime.now() - start_time).total_seconds()
                
                # Second call - should use cache
                start_time = datetime.now()
                financial_data2 = await manager.get_financial_data()
                second_call_time = (datetime.now() - start_time).total_seconds()
                
                if financial_data1 and financial_data2:
                    cache_working = second_call_time < first_call_time
                    self.log_test_result(
                        "Performance - Caching Mechanism",
                        cache_working,
                        f"First call: {first_call_time:.3f}s, Second call: {second_call_time:.3f}s"
                    )
                else:
                    self.log_test_result("Performance - Caching Mechanism", False, "No data retrieved")
                    
        except Exception as e:
            self.log_test_result("Performance Testing", False, str(e))
            
    def print_summary(self):
        """Print test summary"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        print("=" * 60)
        print("EXTERNAL SYSTEM INTEGRATION TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%")
        print(f"Duration: {duration:.2f} seconds")
        print()
        
        if failed_tests > 0:
            print("Failed Tests:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  ❌ {result['test_name']}: {result['details']}")
                    
        print("=" * 60)
        
        return passed_tests == total_tests


async def main():
    """Main test execution function"""
    print("Starting External System Integration Tests (Phase 8)")
    print("=" * 60)
    print()
    
    tester = ExternalSystemIntegrationTester()
    
    # Run all test suites
    await tester.test_erp_connector()
    await tester.test_crm_connector()
    await tester.test_bi_connector()
    await tester.test_integration_manager()
    await tester.test_cross_system_analytics()
    await tester.test_error_handling_and_fallbacks()
    await tester.test_performance_and_caching()
    
    # Print summary
    success = tester.print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
