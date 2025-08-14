"""
Integration Manager for Decision Support System

This module provides a unified interface for accessing data from multiple
external business systems including ERP, CRM, and BI tools.

Coordinates:
- ERP System Connector (SAP, Oracle, Dynamics, NetSuite, Infor)
- CRM System Connector (Salesforce, HubSpot, Dynamics CRM, Pipedrive, Zoho)
- BI Tool Connector (Tableau, Power BI, Qlik, Looker, Sisense)

Provides unified data access and cross-system analytics.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass

from .erp_connector import ERPConnector, ERPConfig, ERPType, FinancialData, InventoryData, SupplyChainData
from .crm_connector import CRMConnector, CRMConfig, CRMType, CustomerProfile, SalesPipeline, InteractionData
from .bi_connector import BIConnector, BIConfig, BIType, DashboardData, KPIMetric, ReportData

logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuration for all external system integrations"""
    erp_config: Optional[ERPConfig] = None
    crm_config: Optional[CRMConfig] = None
    bi_config: Optional[BIConfig] = None
    cache_duration: int = 300  # 5 minutes
    max_concurrent_requests: int = 10


@dataclass
class UnifiedBusinessData:
    """Unified business data structure combining all systems"""
    financial_data: Optional[FinancialData] = None
    inventory_data: Optional[InventoryData] = None
    supply_chain_data: Optional[SupplyChainData] = None
    customer_profiles: List[CustomerProfile] = None
    sales_pipeline: Optional[SalesPipeline] = None
    customer_interactions: List[InteractionData] = None
    dashboard_data: List[DashboardData] = None
    kpi_metrics: List[KPIMetric] = None
    reports: List[ReportData] = None
    timestamp: Optional[datetime] = None


class IntegrationManager:
    """
    Integration Manager for coordinating external business system connectors
    
    Provides unified access to:
    - ERP system data (financial, inventory, supply chain)
    - CRM system data (customers, sales, interactions)
    - BI system data (dashboards, KPIs, reports)
    - Cross-system analytics and insights
    """
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self._erp_connector: Optional[ERPConnector] = None
        self._crm_connector: Optional[CRMConnector] = None
        self._bi_connector: Optional[BIConnector] = None
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
    async def __aenter__(self):
        """Async context manager entry"""
        # Initialize connectors
        if self.config.erp_config:
            self._erp_connector = ERPConnector(self.config.erp_config)
            await self._erp_connector.__aenter__()
            
        if self.config.crm_config:
            self._crm_connector = CRMConnector(self.config.crm_config)
            await self._crm_connector.__aenter__()
            
        if self.config.bi_config:
            self._bi_connector = BIConnector(self.config.bi_config)
            await self._bi_connector.__aenter__()
            
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._erp_connector:
            await self._erp_connector.__aexit__(exc_type, exc_val, exc_tb)
        if self._crm_connector:
            await self._crm_connector.__aexit__(exc_type, exc_val, exc_tb)
        if self._bi_connector:
            await self._bi_connector.__aexit__(exc_type, exc_val, exc_tb)
            
    def _get_cache_key(self, system: str, method: str, *args) -> str:
        """Generate cache key for method calls"""
        return f"{system}:{method}:{':'.join(str(arg) for arg in args)}"
        
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self._cache_timestamps:
            return False
        return (datetime.now() - self._cache_timestamps[cache_key]).seconds < self.config.cache_duration
        
    def _set_cache(self, cache_key: str, data: Any):
        """Set cache data with timestamp"""
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = datetime.now()
        
    def _get_cache(self, cache_key: str) -> Optional[Any]:
        """Get cached data if valid"""
        if self._is_cache_valid(cache_key):
            return self._cache.get(cache_key)
        return None
        
    async def get_unified_business_data(self, customer_ids: Optional[List[str]] = None) -> UnifiedBusinessData:
        """Get unified business data from all connected systems"""
        async with self._semaphore:
            try:
                # Get ERP data
                financial_data = await self.get_financial_data()
                inventory_data = await self.get_inventory_data()
                supply_chain_data = await self.get_supply_chain_data()
                
                # Get CRM data
                customer_profiles = []
                sales_pipeline = await self.get_sales_pipeline()
                customer_interactions = []
                
                if customer_ids:
                    for customer_id in customer_ids:
                        profile = await self.get_customer_profile(customer_id)
                        if profile:
                            customer_profiles.append(profile)
                        interactions = await self.get_customer_interactions(customer_id)
                        customer_interactions.extend(interactions)
                
                # Get BI data
                dashboard_data = await self.get_dashboard_data()
                kpi_metrics = await self.get_kpi_metrics()
                reports = await self.get_reports()
                
                return UnifiedBusinessData(
                    financial_data=financial_data,
                    inventory_data=inventory_data,
                    supply_chain_data=supply_chain_data,
                    customer_profiles=customer_profiles,
                    sales_pipeline=sales_pipeline,
                    customer_interactions=customer_interactions,
                    dashboard_data=dashboard_data,
                    kpi_metrics=kpi_metrics,
                    reports=reports,
                    timestamp=datetime.now()
                )
                
            except Exception as e:
                logger.error(f"Error getting unified business data: {e}")
                return UnifiedBusinessData(timestamp=datetime.now())
                
    # ERP System Methods
    async def get_financial_data(self, period: str = "monthly", currency: str = "USD") -> Optional[FinancialData]:
        """Get financial data from ERP system"""
        if not self._erp_connector:
            return None
            
        cache_key = self._get_cache_key("erp", "financial", period, currency)
        cached_data = self._get_cache(cache_key)
        if cached_data:
            return cached_data
            
        try:
            data = await self._erp_connector.get_financial_data(period, currency)
            self._set_cache(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"Error getting financial data: {e}")
            return None
            
    async def get_inventory_data(self) -> Optional[InventoryData]:
        """Get inventory data from ERP system"""
        if not self._erp_connector:
            return None
            
        cache_key = self._get_cache_key("erp", "inventory")
        cached_data = self._get_cache(cache_key)
        if cached_data:
            return cached_data
            
        try:
            data = await self._erp_connector.get_inventory_data()
            self._set_cache(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"Error getting inventory data: {e}")
            return None
            
    async def get_supply_chain_data(self) -> Optional[SupplyChainData]:
        """Get supply chain data from ERP system"""
        if not self._erp_connector:
            return None
            
        cache_key = self._get_cache_key("erp", "supply_chain")
        cached_data = self._get_cache(cache_key)
        if cached_data:
            return cached_data
            
        try:
            data = await self._erp_connector.get_supply_chain_data()
            self._set_cache(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"Error getting supply chain data: {e}")
            return None
            
    async def get_customer_data(self, customer_id: Optional[str] = None) -> Dict[str, Any]:
        """Get customer data from ERP system"""
        if not self._erp_connector:
            return {}
            
        cache_key = self._get_cache_key("erp", "customer", customer_id or "all")
        cached_data = self._get_cache(cache_key)
        if cached_data:
            return cached_data
            
        try:
            data = await self._erp_connector.get_customer_data(customer_id)
            self._set_cache(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"Error getting customer data: {e}")
            return {}
            
    async def get_production_metrics(self) -> Dict[str, Any]:
        """Get production metrics from ERP system"""
        if not self._erp_connector:
            return {}
            
        cache_key = self._get_cache_key("erp", "production")
        cached_data = self._get_cache(cache_key)
        if cached_data:
            return cached_data
            
        try:
            data = await self._erp_connector.get_production_metrics()
            self._set_cache(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"Error getting production metrics: {e}")
            return {}
            
    # CRM System Methods
    async def get_customer_profile(self, customer_id: str) -> Optional[CustomerProfile]:
        """Get customer profile from CRM system"""
        if not self._crm_connector:
            return None
            
        cache_key = self._get_cache_key("crm", "profile", customer_id)
        cached_data = self._get_cache(cache_key)
        if cached_data:
            return cached_data
            
        try:
            data = await self._crm_connector.get_customer_profile(customer_id)
            self._set_cache(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"Error getting customer profile: {e}")
            return None
            
    async def get_sales_pipeline(self) -> Optional[SalesPipeline]:
        """Get sales pipeline from CRM system"""
        if not self._crm_connector:
            return None
            
        cache_key = self._get_cache_key("crm", "pipeline")
        cached_data = self._get_cache(cache_key)
        if cached_data:
            return cached_data
            
        try:
            data = await self._crm_connector.get_sales_pipeline()
            self._set_cache(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"Error getting sales pipeline: {e}")
            return None
            
    async def get_customer_interactions(self, customer_id: str, days: int = 30) -> List[InteractionData]:
        """Get customer interactions from CRM system"""
        if not self._crm_connector:
            return []
            
        cache_key = self._get_cache_key("crm", "interactions", customer_id, days)
        cached_data = self._get_cache(cache_key)
        if cached_data:
            return cached_data
            
        try:
            data = await self._crm_connector.get_customer_interactions(customer_id, days)
            self._set_cache(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"Error getting customer interactions: {e}")
            return []
            
    async def get_lead_data(self, lead_id: Optional[str] = None) -> Dict[str, Any]:
        """Get lead data from CRM system"""
        if not self._crm_connector:
            return {}
            
        cache_key = self._get_cache_key("crm", "lead", lead_id or "all")
        cached_data = self._get_cache(cache_key)
        if cached_data:
            return cached_data
            
        try:
            data = await self._crm_connector.get_lead_data(lead_id)
            self._set_cache(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"Error getting lead data: {e}")
            return {}
            
    async def get_sales_performance(self) -> Dict[str, Any]:
        """Get sales performance from CRM system"""
        if not self._crm_connector:
            return {}
            
        cache_key = self._get_cache_key("crm", "performance")
        cached_data = self._get_cache(cache_key)
        if cached_data:
            return cached_data
            
        try:
            data = await self._crm_connector.get_sales_performance()
            self._set_cache(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"Error getting sales performance: {e}")
            return {}
            
    # BI System Methods
    async def get_dashboard_data(self, dashboard_ids: Optional[List[str]] = None) -> List[DashboardData]:
        """Get dashboard data from BI system"""
        if not self._bi_connector:
            return []
            
        if not dashboard_ids:
            dashboard_ids = ["default"]
            
        results = []
        for dashboard_id in dashboard_ids:
            cache_key = self._get_cache_key("bi", "dashboard", dashboard_id)
            cached_data = self._get_cache(cache_key)
            if cached_data:
                results.append(cached_data)
            else:
                try:
                    data = await self._bi_connector.get_dashboard_data(dashboard_id)
                    if data:
                        self._set_cache(cache_key, data)
                        results.append(data)
                except Exception as e:
                    logger.error(f"Error getting dashboard data: {e}")
                    
        return results
        
    async def get_kpi_metrics(self, category: Optional[str] = None) -> List[KPIMetric]:
        """Get KPI metrics from BI system"""
        if not self._bi_connector:
            return []
            
        cache_key = self._get_cache_key("bi", "kpi", category or "all")
        cached_data = self._get_cache(cache_key)
        if cached_data:
            return cached_data
            
        try:
            data = await self._bi_connector.get_kpi_metrics(category)
            self._set_cache(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"Error getting KPI metrics: {e}")
            return []
            
    async def get_reports(self, report_ids: Optional[List[str]] = None) -> List[ReportData]:
        """Get reports from BI system"""
        if not self._bi_connector:
            return []
            
        if not report_ids:
            report_ids = ["default"]
            
        results = []
        for report_id in report_ids:
            cache_key = self._get_cache_key("bi", "report", report_id)
            cached_data = self._get_cache(cache_key)
            if cached_data:
                results.append(cached_data)
            else:
                try:
                    data = await self._bi_connector.get_report_data(report_id)
                    if data:
                        self._set_cache(cache_key, data)
                        results.append(data)
                except Exception as e:
                    logger.error(f"Error getting report data: {e}")
                    
        return results
        
    async def execute_bi_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a custom query on the BI system"""
        if not self._bi_connector:
            return {}
            
        cache_key = self._get_cache_key("bi", "query", query, str(parameters or {}))
        cached_data = self._get_cache(cache_key)
        if cached_data:
            return cached_data
            
        try:
            data = await self._bi_connector.execute_query(query, parameters)
            self._set_cache(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"Error executing BI query: {e}")
            return {}
            
    async def get_data_warehouse_info(self) -> Dict[str, Any]:
        """Get data warehouse information from BI system"""
        if not self._bi_connector:
            return {}
            
        cache_key = self._get_cache_key("bi", "warehouse")
        cached_data = self._get_cache(cache_key)
        if cached_data:
            return cached_data
            
        try:
            data = await self._bi_connector.get_data_warehouse_info()
            self._set_cache(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"Error getting data warehouse info: {e}")
            return {}
            
    # Cross-System Analytics Methods
    async def get_customer_360_view(self, customer_id: str) -> Dict[str, Any]:
        """Get comprehensive 360-degree view of a customer across all systems"""
        try:
            # Get customer data from all systems
            erp_customer = await self.get_customer_data(customer_id)
            crm_profile = await self.get_customer_profile(customer_id)
            crm_interactions = await self.get_customer_interactions(customer_id)
            
            # Get related business data
            financial_data = await self.get_financial_data()
            sales_pipeline = await self.get_sales_pipeline()
            
            return {
                "customer_id": customer_id,
                "erp_data": erp_customer,
                "crm_profile": crm_profile,
                "interactions": crm_interactions,
                "financial_context": financial_data,
                "sales_context": sales_pipeline,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting customer 360 view: {e}")
            return {"customer_id": customer_id, "error": str(e)}
            
    async def get_business_health_score(self) -> Dict[str, Any]:
        """Calculate overall business health score from all systems"""
        try:
            # Get data from all systems
            financial_data = await self.get_financial_data()
            inventory_data = await self.get_inventory_data()
            supply_chain_data = await self.get_supply_chain_data()
            sales_pipeline = await self.get_sales_pipeline()
            kpi_metrics = await self.get_kpi_metrics()
            
            # Calculate health scores
            financial_score = self._calculate_financial_health(financial_data)
            operational_score = self._calculate_operational_health(inventory_data, supply_chain_data)
            sales_score = self._calculate_sales_health(sales_pipeline)
            kpi_score = self._calculate_kpi_health(kpi_metrics)
            
            # Overall weighted score
            overall_score = (
                financial_score * 0.3 +
                operational_score * 0.25 +
                sales_score * 0.25 +
                kpi_score * 0.2
            )
            
            return {
                "overall_score": overall_score,
                "financial_health": financial_score,
                "operational_health": operational_score,
                "sales_health": sales_score,
                "kpi_health": kpi_score,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error calculating business health score: {e}")
            return {"error": str(e)}
            
    def _calculate_financial_health(self, financial_data: Optional[FinancialData]) -> float:
        """Calculate financial health score"""
        if not financial_data:
            return 0.5
            
        # Simple scoring based on profit margin and cash flow
        profit_score = min(financial_data.profit_margin * 100, 100)
        cash_flow_score = min((financial_data.cash_flow / financial_data.revenue) * 100, 100)
        
        return (profit_score + cash_flow_score) / 2
        
    def _calculate_operational_health(self, inventory_data: Optional[InventoryData], 
                                    supply_chain_data: Optional[SupplyChainData]) -> float:
        """Calculate operational health score"""
        if not inventory_data or not supply_chain_data:
            return 0.5
            
        # Inventory health
        inventory_score = 100 - (inventory_data.out_of_stock_items / inventory_data.total_items * 100)
        
        # Supply chain health
        avg_supplier_performance = sum(supply_chain_data.supplier_performance.values()) / len(supply_chain_data.supplier_performance) if supply_chain_data.supplier_performance else 0.5
        supply_chain_score = avg_supplier_performance * 100
        
        return (inventory_score + supply_chain_score) / 2
        
    def _calculate_sales_health(self, sales_pipeline: Optional[SalesPipeline]) -> float:
        """Calculate sales health score"""
        if not sales_pipeline:
            return 0.5
            
        # Conversion rate and pipeline velocity
        conversion_score = sales_pipeline.conversion_rate * 100
        velocity_score = min(100, (sales_pipeline.total_leads / sales_pipeline.sales_cycle_days) * 10)
        
        return (conversion_score + velocity_score) / 2
        
    def _calculate_kpi_health(self, kpi_metrics: List[KPIMetric]) -> float:
        """Calculate KPI health score"""
        if not kpi_metrics:
            return 0.5
            
        # Average KPI performance against targets
        scores = []
        for metric in kpi_metrics:
            if metric.target:
                performance = (metric.value / metric.target) * 100
                scores.append(min(performance, 100))
            else:
                scores.append(75)  # Default score for metrics without targets
                
        return sum(scores) / len(scores) if scores else 0.5
        
    async def clear_cache(self):
        """Clear all cached data"""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Integration manager cache cleared")
        
    async def get_system_status(self) -> Dict[str, Any]:
        """Get status of all connected systems"""
        status = {
            "erp_connected": self._erp_connector is not None,
            "crm_connected": self._crm_connector is not None,
            "bi_connected": self._bi_connector is not None,
            "cache_size": len(self._cache),
            "timestamp": datetime.now().isoformat()
        }
        
        return status
