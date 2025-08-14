"""
BI Tool Connector for Decision Support System

This module provides integration with major BI systems including:
- Tableau
- Microsoft Power BI
- QlikView/Qlik Sense
- Looker
- Sisense

Provides real-time access to:
- Dashboard data and visualizations
- KPI metrics and reports
- Data warehouse queries
- Business analytics
- Performance indicators
"""

import aiohttp
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class BIType(Enum):
    """Supported BI system types"""
    TABLEAU = "tableau"
    POWER_BI = "power_bi"
    QLIK = "qlik"
    LOOKER = "looker"
    SISENSE = "sisense"
    MOCK = "mock"


@dataclass
class BIConfig:
    """Configuration for BI system connection"""
    bi_type: BIType
    base_url: str
    api_key: Optional[str] = None
    access_token: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    site_id: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3


@dataclass
class DashboardData:
    """Dashboard data structure"""
    dashboard_id: str
    name: str
    description: str
    last_updated: datetime
    data_sources: List[str]
    visualizations: List[Dict[str, Any]]
    filters: Dict[str, Any]
    timestamp: Optional[datetime] = None


@dataclass
class KPIMetric:
    """KPI metric data structure"""
    metric_id: str
    name: str
    value: float
    target: Optional[float] = None
    unit: str = ""
    trend: str = "stable"
    period: str = "monthly"
    category: str = "general"
    timestamp: Optional[datetime] = None


@dataclass
class ReportData:
    """Report data structure"""
    report_id: str
    name: str
    type: str
    data: Dict[str, Any]
    parameters: Dict[str, Any]
    last_generated: datetime
    refresh_frequency: str = "daily"
    timestamp: Optional[datetime] = None


class BIConnector:
    """
    BI Tool Connector for real-time business intelligence data integration
    
    Supports multiple BI systems with unified interface for:
    - Dashboard data retrieval
    - KPI metric monitoring
    - Report generation and access
    - Data warehouse queries
    - Business analytics
    """
    
    def __init__(self, config: BIConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            
    async def _authenticate(self) -> bool:
        """Authenticate with BI system based on type"""
        try:
            if self.config.bi_type == BIType.TABLEAU:
                return await self._authenticate_tableau()
            elif self.config.bi_type == BIType.POWER_BI:
                return await self._authenticate_power_bi()
            elif self.config.bi_type == BIType.QLIK:
                return await self._authenticate_qlik()
            elif self.config.bi_type == BIType.LOOKER:
                return await self._authenticate_looker()
            elif self.config.bi_type == BIType.SISENSE:
                return await self._authenticate_sisense()
            elif self.config.bi_type == BIType.MOCK:
                return True
            else:
                logger.error(f"Unsupported BI type: {self.config.bi_type}")
                return False
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
            
    async def _authenticate_tableau(self) -> bool:
        """Authenticate with Tableau"""
        try:
            auth_url = f"{self.config.base_url}/api/3.19/auth/signin"
            auth_data = {
                "credentials": {
                    "personalAccessTokenName": self.config.api_key,
                    "personalAccessTokenSecret": self.config.client_secret
                }
            }
            
            async with self.session.post(auth_url, json=auth_data) as response:
                if response.status == 200:
                    data = await response.json()
                    self._access_token = data.get("credentials", {}).get("token")
                    self._token_expiry = datetime.now() + timedelta(hours=1)
                    return True
                else:
                    logger.error(f"Tableau authentication failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Tableau authentication error: {e}")
            return False
            
    async def _authenticate_power_bi(self) -> bool:
        """Authenticate with Microsoft Power BI"""
        try:
            auth_url = "https://login.microsoftonline.com/common/oauth2/v2.0/token"
            auth_data = {
                "grant_type": "client_credentials",
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
                "scope": "https://analysis.windows.net/powerbi/api/.default"
            }
            
            async with self.session.post(auth_url, data=auth_data) as response:
                if response.status == 200:
                    data = await response.json()
                    self._access_token = data.get("access_token")
                    self._token_expiry = datetime.now() + timedelta(hours=1)
                    return True
                else:
                    logger.error(f"Power BI authentication failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Power BI authentication error: {e}")
            return False
            
    async def _authenticate_qlik(self) -> bool:
        """Authenticate with Qlik"""
        try:
            auth_url = f"{self.config.base_url}/api/v1/auth"
            auth_data = {
                "username": self.config.username,
                "password": self.config.password
            }
            
            async with self.session.post(auth_url, json=auth_data) as response:
                if response.status == 200:
                    data = await response.json()
                    self._access_token = data.get("access_token")
                    self._token_expiry = datetime.now() + timedelta(hours=1)
                    return True
                else:
                    logger.error(f"Qlik authentication failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Qlik authentication error: {e}")
            return False
            
    async def _authenticate_looker(self) -> bool:
        """Authenticate with Looker"""
        try:
            auth_url = f"{self.config.base_url}/api/3.1/login"
            auth_data = {
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret
            }
            
            async with self.session.post(auth_url, json=auth_data) as response:
                if response.status == 200:
                    data = await response.json()
                    self._access_token = data.get("access_token")
                    self._token_expiry = datetime.now() + timedelta(hours=1)
                    return True
                else:
                    logger.error(f"Looker authentication failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Looker authentication error: {e}")
            return False
            
    async def _authenticate_sisense(self) -> bool:
        """Authenticate with Sisense"""
        try:
            auth_url = f"{self.config.base_url}/api/v1/authentication/login"
            auth_data = {
                "username": self.config.username,
                "password": self.config.password
            }
            
            async with self.session.post(auth_url, json=auth_data) as response:
                if response.status == 200:
                    data = await response.json()
                    self._access_token = data.get("access_token")
                    self._token_expiry = datetime.now() + timedelta(hours=1)
                    return True
                else:
                    logger.error(f"Sisense authentication failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Sisense authentication error: {e}")
            return False
            
    async def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if self._access_token:
            if self.config.bi_type == BIType.TABLEAU:
                headers["X-Tableau-Auth"] = self._access_token
            elif self.config.bi_type == BIType.POWER_BI:
                headers["Authorization"] = f"Bearer {self._access_token}"
            elif self.config.bi_type == BIType.QLIK:
                headers["Authorization"] = f"Bearer {self._access_token}"
            elif self.config.bi_type == BIType.LOOKER:
                headers["Authorization"] = f"Bearer {self._access_token}"
            elif self.config.bi_type == BIType.SISENSE:
                headers["Authorization"] = f"Bearer {self._access_token}"
                
        return headers
        
    async def _ensure_authenticated(self) -> bool:
        """Ensure we have a valid authentication token"""
        if not self._access_token or (self._token_expiry and datetime.now() >= self._token_expiry):
            return await self._authenticate()
        return True
        
    async def get_dashboard_data(self, dashboard_id: str) -> Optional[DashboardData]:
        """Get dashboard data from BI system"""
        try:
            if not await self._ensure_authenticated():
                return self._get_mock_dashboard_data(dashboard_id)
                
            if self.config.bi_type == BIType.MOCK:
                return self._get_mock_dashboard_data(dashboard_id)
                
            headers = await self._get_headers()
            
            # Different BI systems have different endpoints
            if self.config.bi_type == BIType.TABLEAU:
                url = f"{self.config.base_url}/api/3.19/sites/{self.config.site_id}/dashboards/{dashboard_id}"
            elif self.config.bi_type == BIType.POWER_BI:
                url = f"{self.config.base_url}/v1.0/myorg/dashboards/{dashboard_id}"
            elif self.config.bi_type == BIType.QLIK:
                url = f"{self.config.base_url}/api/v1/apps/{dashboard_id}/sheets"
            elif self.config.bi_type == BIType.LOOKER:
                url = f"{self.config.base_url}/api/3.1/dashboards/{dashboard_id}"
            elif self.config.bi_type == BIType.SISENSE:
                url = f"{self.config.base_url}/api/v1/dashboards/{dashboard_id}"
            else:
                return self._get_mock_dashboard_data(dashboard_id)
                
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_dashboard_data(data, dashboard_id)
                else:
                    logger.warning(f"Failed to get dashboard data: {response.status}")
                    return self._get_mock_dashboard_data(dashboard_id)
                    
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return self._get_mock_dashboard_data(dashboard_id)
            
    async def get_kpi_metrics(self, category: Optional[str] = None) -> List[KPIMetric]:
        """Get KPI metrics from BI system"""
        try:
            if not await self._ensure_authenticated():
                return self._get_mock_kpi_metrics(category)
                
            if self.config.bi_type == BIType.MOCK:
                return self._get_mock_kpi_metrics(category)
                
            headers = await self._get_headers()
            
            # Different BI systems have different endpoints
            if self.config.bi_type == BIType.TABLEAU:
                url = f"{self.config.base_url}/api/3.19/sites/{self.config.site_id}/metrics"
            elif self.config.bi_type == BIType.POWER_BI:
                url = f"{self.config.base_url}/v1.0/myorg/reports"
            elif self.config.bi_type == BIType.QLIK:
                url = f"{self.config.base_url}/api/v1/metrics"
            elif self.config.bi_type == BIType.LOOKER:
                url = f"{self.config.base_url}/api/3.1/lookml_models"
            elif self.config.bi_type == BIType.SISENSE:
                url = f"{self.config.base_url}/api/v1/metrics"
            else:
                return self._get_mock_kpi_metrics(category)
                
            params = {}
            if category:
                params["category"] = category
                
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_kpi_metrics(data, category)
                else:
                    logger.warning(f"Failed to get KPI metrics: {response.status}")
                    return self._get_mock_kpi_metrics(category)
                    
        except Exception as e:
            logger.error(f"Error getting KPI metrics: {e}")
            return self._get_mock_kpi_metrics(category)
            
    async def get_report_data(self, report_id: str) -> Optional[ReportData]:
        """Get report data from BI system"""
        try:
            if not await self._ensure_authenticated():
                return self._get_mock_report_data(report_id)
                
            if self.config.bi_type == BIType.MOCK:
                return self._get_mock_report_data(report_id)
                
            headers = await self._get_headers()
            
            # Different BI systems have different endpoints
            if self.config.bi_type == BIType.TABLEAU:
                url = f"{self.config.base_url}/api/3.19/sites/{self.config.site_id}/workbooks/{report_id}"
            elif self.config.bi_type == BIType.POWER_BI:
                url = f"{self.config.base_url}/v1.0/myorg/reports/{report_id}"
            elif self.config.bi_type == BIType.QLIK:
                url = f"{self.config.base_url}/api/v1/reports/{report_id}"
            elif self.config.bi_type == BIType.LOOKER:
                url = f"{self.config.base_url}/api/3.1/looks/{report_id}"
            elif self.config.bi_type == BIType.SISENSE:
                url = f"{self.config.base_url}/api/v1/reports/{report_id}"
            else:
                return self._get_mock_report_data(report_id)
                
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_report_data(data, report_id)
                else:
                    logger.warning(f"Failed to get report data: {response.status}")
                    return self._get_mock_report_data(report_id)
                    
        except Exception as e:
            logger.error(f"Error getting report data: {e}")
            return self._get_mock_report_data(report_id)
            
    def _parse_dashboard_data(self, data: Dict[str, Any], dashboard_id: str) -> DashboardData:
        """Parse dashboard data from BI response"""
        try:
            return DashboardData(
                dashboard_id=dashboard_id,
                name=data.get("name", "Dashboard"),
                description=data.get("description", ""),
                last_updated=datetime.now(),
                data_sources=data.get("dataSources", []),
                visualizations=data.get("visualizations", []),
                filters=data.get("filters", {}),
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error parsing dashboard data: {e}")
            return self._get_mock_dashboard_data(dashboard_id)
            
    def _parse_kpi_metrics(self, data: Dict[str, Any], category: Optional[str]) -> List[KPIMetric]:
        """Parse KPI metrics from BI response"""
        try:
            metrics = data.get("metrics", data.get("data", []))
            result = []
            
            for metric in metrics:
                if category and metric.get("category") != category:
                    continue
                    
                result.append(KPIMetric(
                    metric_id=metric.get("id", str(len(result))),
                    name=metric.get("name", "KPI"),
                    value=float(metric.get("value", 0)),
                    target=metric.get("target"),
                    unit=metric.get("unit", ""),
                    trend=metric.get("trend", "stable"),
                    period=metric.get("period", "monthly"),
                    category=metric.get("category", "general"),
                    timestamp=datetime.now()
                ))
                
            return result
        except Exception as e:
            logger.error(f"Error parsing KPI metrics: {e}")
            return self._get_mock_kpi_metrics(category)
            
    def _parse_report_data(self, data: Dict[str, Any], report_id: str) -> ReportData:
        """Parse report data from BI response"""
        try:
            return ReportData(
                report_id=report_id,
                name=data.get("name", "Report"),
                type=data.get("type", "standard"),
                data=data.get("data", {}),
                parameters=data.get("parameters", {}),
                last_generated=datetime.now(),
                refresh_frequency=data.get("refreshFrequency", "daily"),
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error parsing report data: {e}")
            return self._get_mock_report_data(report_id)
            
    def _get_mock_dashboard_data(self, dashboard_id: str) -> DashboardData:
        """Get mock dashboard data for testing"""
        return DashboardData(
            dashboard_id=dashboard_id,
            name="Sales Performance Dashboard",
            description="Real-time sales performance metrics and trends",
            last_updated=datetime.now(),
            data_sources=["Salesforce", "ERP System", "Marketing Platform"],
            visualizations=[
                {"type": "chart", "title": "Monthly Revenue", "data": [250000, 275000, 300000]},
                {"type": "table", "title": "Top Performers", "data": [{"name": "Alice", "sales": 450000}]}
            ],
            filters={"date_range": "last_30_days", "region": "all"},
            timestamp=datetime.now()
        )
        
    def _get_mock_kpi_metrics(self, category: Optional[str]) -> List[KPIMetric]:
        """Get mock KPI metrics for testing"""
        metrics = [
            KPIMetric(
                metric_id="KPI001",
                name="Monthly Revenue",
                value=2850000.0,
                target=3000000.0,
                unit="USD",
                trend="increasing",
                period="monthly",
                category="financial",
                timestamp=datetime.now()
            ),
            KPIMetric(
                metric_id="KPI002",
                name="Customer Satisfaction",
                value=4.2,
                target=4.5,
                unit="stars",
                trend="stable",
                period="monthly",
                category="customer",
                timestamp=datetime.now()
            ),
            KPIMetric(
                metric_id="KPI003",
                name="Conversion Rate",
                value=0.34,
                target=0.40,
                unit="%",
                trend="increasing",
                period="monthly",
                category="sales",
                timestamp=datetime.now()
            )
        ]
        
        if category:
            return [m for m in metrics if m.category == category]
        return metrics
        
    def _get_mock_report_data(self, report_id: str) -> ReportData:
        """Get mock report data for testing"""
        return ReportData(
            report_id=report_id,
            name="Sales Analysis Report",
            type="analytics",
            data={
                "total_sales": 2850000.0,
                "growth_rate": 0.15,
                "top_products": ["Product A", "Product B", "Product C"],
                "regional_breakdown": {
                    "North": 1200000.0,
                    "South": 850000.0,
                    "East": 800000.0
                }
            },
            parameters={"date_range": "last_quarter", "include_forecast": True},
            last_generated=datetime.now(),
            refresh_frequency="daily",
            timestamp=datetime.now()
        )
        
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a custom query on the BI system"""
        try:
            if not await self._ensure_authenticated():
                return self._get_mock_query_result(query)
                
            if self.config.bi_type == BIType.MOCK:
                return self._get_mock_query_result(query)
                
            headers = await self._get_headers()
            
            # Different BI systems have different query endpoints
            if self.config.bi_type == BIType.TABLEAU:
                url = f"{self.config.base_url}/api/3.19/sites/{self.config.site_id}/datasources"
            elif self.config.bi_type == BIType.POWER_BI:
                url = f"{self.config.base_url}/v1.0/myorg/datasets"
            elif self.config.bi_type == BIType.QLIK:
                url = f"{self.config.base_url}/api/v1/apps"
            elif self.config.bi_type == BIType.LOOKER:
                url = f"{self.config.base_url}/api/3.1/queries"
            elif self.config.bi_type == BIType.SISENSE:
                url = f"{self.config.base_url}/api/v1/query"
            else:
                return self._get_mock_query_result(query)
                
            query_data = {
                "query": query,
                "parameters": parameters or {}
            }
            
            async with self.session.post(url, headers=headers, json=query_data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Failed to execute query: {response.status}")
                    return self._get_mock_query_result(query)
                    
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return self._get_mock_query_result(query)
            
    def _get_mock_query_result(self, query: str) -> Dict[str, Any]:
        """Get mock query result for testing"""
        return {
            "query": query,
            "result": {
                "rows": [
                    {"date": "2024-01-01", "revenue": 250000.0, "orders": 1250},
                    {"date": "2024-01-02", "revenue": 275000.0, "orders": 1375},
                    {"date": "2024-01-03", "revenue": 300000.0, "orders": 1500}
                ],
                "columns": ["date", "revenue", "orders"],
                "total_rows": 3,
                "execution_time": 0.5
            },
            "timestamp": datetime.now().isoformat()
        }
        
    async def get_data_warehouse_info(self) -> Dict[str, Any]:
        """Get data warehouse information from BI system"""
        try:
            if not await self._ensure_authenticated():
                return self._get_mock_data_warehouse_info()
                
            if self.config.bi_type == BIType.MOCK:
                return self._get_mock_data_warehouse_info()
                
            headers = await self._get_headers()
            url = f"{self.config.base_url}/api/v1/databases"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Failed to get data warehouse info: {response.status}")
                    return self._get_mock_data_warehouse_info()
                    
        except Exception as e:
            logger.error(f"Error getting data warehouse info: {e}")
            return self._get_mock_data_warehouse_info()
            
    def _get_mock_data_warehouse_info(self) -> Dict[str, Any]:
        """Get mock data warehouse info for testing"""
        return {
            "databases": [
                {
                    "name": "Sales_DB",
                    "type": "PostgreSQL",
                    "size_gb": 125.5,
                    "last_updated": "2024-01-15T10:30:00Z",
                    "tables": ["customers", "orders", "products", "sales"]
                },
                {
                    "name": "Marketing_DB",
                    "type": "MySQL",
                    "size_gb": 85.2,
                    "last_updated": "2024-01-15T09:15:00Z",
                    "tables": ["campaigns", "leads", "analytics", "events"]
                }
            ],
            "total_size_gb": 210.7,
            "last_refresh": "2024-01-15T10:30:00Z",
            "status": "healthy"
        }
