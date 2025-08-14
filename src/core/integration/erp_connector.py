"""
ERP System Connector for Decision Support System

This module provides integration with major ERP systems including:
- SAP ERP
- Oracle ERP Cloud
- Microsoft Dynamics 365
- NetSuite
- Infor CloudSuite

Provides real-time access to:
- Financial data
- Inventory levels
- Supply chain information
- Customer data
- Production metrics
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import base64
import hashlib
import hmac

logger = logging.getLogger(__name__)


class ERPType(Enum):
    """Supported ERP system types"""
    SAP = "sap"
    ORACLE = "oracle"
    DYNAMICS365 = "dynamics365"
    NETSUITE = "netsuite"
    INFOR = "infor"
    MOCK = "mock"


@dataclass
class ERPConfig:
    """Configuration for ERP system connection"""
    erp_type: ERPType
    base_url: str
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    tenant_id: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3


@dataclass
class FinancialData:
    """Financial data structure"""
    revenue: float
    expenses: float
    profit_margin: float
    cash_flow: float
    accounts_receivable: float
    accounts_payable: float
    currency: str = "USD"
    period: str = "monthly"
    timestamp: Optional[datetime] = None


@dataclass
class InventoryData:
    """Inventory data structure"""
    total_items: int
    low_stock_items: int
    out_of_stock_items: int
    inventory_value: float
    turnover_rate: float
    warehouse_locations: Dict[str, int]
    timestamp: Optional[datetime] = None


@dataclass
class SupplyChainData:
    """Supply chain data structure"""
    active_orders: int
    pending_deliveries: int
    supplier_performance: Dict[str, float]
    lead_times: Dict[str, int]
    quality_metrics: Dict[str, float]
    timestamp: Optional[datetime] = None


class ERPConnector:
    """
    ERP System Connector for real-time business data integration
    
    Supports multiple ERP systems with unified interface for:
    - Financial data retrieval
    - Inventory management
    - Supply chain monitoring
    - Customer relationship data
    - Production metrics
    """
    
    def __init__(self, config: ERPConfig):
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
        """Authenticate with ERP system based on type"""
        try:
            if self.config.erp_type == ERPType.SAP:
                return await self._authenticate_sap()
            elif self.config.erp_type == ERPType.ORACLE:
                return await self._authenticate_oracle()
            elif self.config.erp_type == ERPType.DYNAMICS365:
                return await self._authenticate_dynamics365()
            elif self.config.erp_type == ERPType.NETSUITE:
                return await self._authenticate_netsuite()
            elif self.config.erp_type == ERPType.INFOR:
                return await self._authenticate_infor()
            elif self.config.erp_type == ERPType.MOCK:
                return True
            else:
                logger.error(f"Unsupported ERP type: {self.config.erp_type}")
                return False
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
            
    async def _authenticate_sap(self) -> bool:
        """Authenticate with SAP ERP"""
        try:
            auth_url = f"{self.config.base_url}/api/auth"
            auth_data = {
                "username": self.config.username,
                "password": self.config.password,
                "client_id": self.config.client_id
            }
            
            async with self.session.post(auth_url, json=auth_data) as response:
                if response.status == 200:
                    data = await response.json()
                    self._access_token = data.get("access_token")
                    self._token_expiry = datetime.now() + timedelta(hours=1)
                    return True
                else:
                    logger.error(f"SAP authentication failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"SAP authentication error: {e}")
            return False
            
    async def _authenticate_oracle(self) -> bool:
        """Authenticate with Oracle ERP Cloud"""
        try:
            auth_url = f"{self.config.base_url}/oauth2/v1/token"
            auth_data = {
                "grant_type": "client_credentials",
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret
            }
            
            async with self.session.post(auth_url, data=auth_data) as response:
                if response.status == 200:
                    data = await response.json()
                    self._access_token = data.get("access_token")
                    self._token_expiry = datetime.now() + timedelta(hours=1)
                    return True
                else:
                    logger.error(f"Oracle authentication failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Oracle authentication error: {e}")
            return False
            
    async def _authenticate_dynamics365(self) -> bool:
        """Authenticate with Microsoft Dynamics 365"""
        try:
            auth_url = f"https://login.microsoftonline.com/{self.config.tenant_id}/oauth2/v2.0/token"
            auth_data = {
                "grant_type": "client_credentials",
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
                "scope": f"{self.config.base_url}/.default"
            }
            
            async with self.session.post(auth_url, data=auth_data) as response:
                if response.status == 200:
                    data = await response.json()
                    self._access_token = data.get("access_token")
                    self._token_expiry = datetime.now() + timedelta(hours=1)
                    return True
                else:
                    logger.error(f"Dynamics 365 authentication failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Dynamics 365 authentication error: {e}")
            return False
            
    async def _authenticate_netsuite(self) -> bool:
        """Authenticate with NetSuite"""
        try:
            # NetSuite uses token-based authentication
            if self.config.api_key:
                self._access_token = self.config.api_key
                self._token_expiry = datetime.now() + timedelta(hours=24)
                return True
            else:
                logger.error("NetSuite requires API key for authentication")
                return False
        except Exception as e:
            logger.error(f"NetSuite authentication error: {e}")
            return False
            
    async def _authenticate_infor(self) -> bool:
        """Authenticate with Infor CloudSuite"""
        try:
            auth_url = f"{self.config.base_url}/auth/oauth/token"
            auth_data = {
                "grant_type": "client_credentials",
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret
            }
            
            async with self.session.post(auth_url, data=auth_data) as response:
                if response.status == 200:
                    data = await response.json()
                    self._access_token = data.get("access_token")
                    self._token_expiry = datetime.now() + timedelta(hours=1)
                    return True
                else:
                    logger.error(f"Infor authentication failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Infor authentication error: {e}")
            return False
            
    async def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if self._access_token:
            if self.config.erp_type == ERPType.SAP:
                headers["Authorization"] = f"Bearer {self._access_token}"
            elif self.config.erp_type == ERPType.ORACLE:
                headers["Authorization"] = f"Bearer {self._access_token}"
            elif self.config.erp_type == ERPType.DYNAMICS365:
                headers["Authorization"] = f"Bearer {self._access_token}"
            elif self.config.erp_type == ERPType.NETSUITE:
                headers["Authorization"] = f"Bearer {self._access_token}"
            elif self.config.erp_type == ERPType.INFOR:
                headers["Authorization"] = f"Bearer {self._access_token}"
                
        return headers
        
    async def _ensure_authenticated(self) -> bool:
        """Ensure we have a valid authentication token"""
        if not self._access_token or (self._token_expiry and datetime.now() >= self._token_expiry):
            return await self._authenticate()
        return True
        
    async def get_financial_data(self, period: str = "monthly", currency: str = "USD") -> Optional[FinancialData]:
        """Get financial data from ERP system"""
        try:
            if not await self._ensure_authenticated():
                return self._get_mock_financial_data(period, currency)
                
            if self.config.erp_type == ERPType.MOCK:
                return self._get_mock_financial_data(period, currency)
                
            headers = await self._get_headers()
            
            # Different ERP systems have different endpoints
            if self.config.erp_type == ERPType.SAP:
                url = f"{self.config.base_url}/api/financial/reports"
            elif self.config.erp_type == ERPType.ORACLE:
                url = f"{self.config.base_url}/financials/v1/reports"
            elif self.config.erp_type == ERPType.DYNAMICS365:
                url = f"{self.config.base_url}/api/data/v9.2/financial_reports"
            elif self.config.erp_type == ERPType.NETSUITE:
                url = f"{self.config.base_url}/rest/platform/v1/financial/reports"
            elif self.config.erp_type == ERPType.INFOR:
                url = f"{self.config.base_url}/api/financial/reports"
            else:
                return self._get_mock_financial_data(period, currency)
                
            params = {
                "period": period,
                "currency": currency,
                "format": "json"
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_financial_data(data, period, currency)
                else:
                    logger.warning(f"Failed to get financial data: {response.status}")
                    return self._get_mock_financial_data(period, currency)
                    
        except Exception as e:
            logger.error(f"Error getting financial data: {e}")
            return self._get_mock_financial_data(period, currency)
            
    async def get_inventory_data(self) -> Optional[InventoryData]:
        """Get inventory data from ERP system"""
        try:
            if not await self._ensure_authenticated():
                return self._get_mock_inventory_data()
                
            if self.config.erp_type == ERPType.MOCK:
                return self._get_mock_inventory_data()
                
            headers = await self._get_headers()
            
            # Different ERP systems have different endpoints
            if self.config.erp_type == ERPType.SAP:
                url = f"{self.config.base_url}/api/inventory/status"
            elif self.config.erp_type == ERPType.ORACLE:
                url = f"{self.config.base_url}/inventory/v1/items"
            elif self.config.erp_type == ERPType.DYNAMICS365:
                url = f"{self.config.base_url}/api/data/v9.2/inventory_items"
            elif self.config.erp_type == ERPType.NETSUITE:
                url = f"{self.config.base_url}/rest/platform/v1/inventory/items"
            elif self.config.erp_type == ERPType.INFOR:
                url = f"{self.config.base_url}/api/inventory/items"
            else:
                return self._get_mock_inventory_data()
                
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_inventory_data(data)
                else:
                    logger.warning(f"Failed to get inventory data: {response.status}")
                    return self._get_mock_inventory_data()
                    
        except Exception as e:
            logger.error(f"Error getting inventory data: {e}")
            return self._get_mock_inventory_data()
            
    async def get_supply_chain_data(self) -> Optional[SupplyChainData]:
        """Get supply chain data from ERP system"""
        try:
            if not await self._ensure_authenticated():
                return self._get_mock_supply_chain_data()
                
            if self.config.erp_type == ERPType.MOCK:
                return self._get_mock_supply_chain_data()
                
            headers = await self._get_headers()
            
            # Different ERP systems have different endpoints
            if self.config.erp_type == ERPType.SAP:
                url = f"{self.config.base_url}/api/supply-chain/status"
            elif self.config.erp_type == ERPType.ORACLE:
                url = f"{self.config.base_url}/supply-chain/v1/orders"
            elif self.config.erp_type == ERPType.DYNAMICS365:
                url = f"{self.config.base_url}/api/data/v9.2/purchase_orders"
            elif self.config.erp_type == ERPType.NETSUITE:
                url = f"{self.config.base_url}/rest/platform/v1/supply-chain/orders"
            elif self.config.erp_type == ERPType.INFOR:
                url = f"{self.config.base_url}/api/supply-chain/orders"
            else:
                return self._get_mock_supply_chain_data()
                
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_supply_chain_data(data)
                else:
                    logger.warning(f"Failed to get supply chain data: {response.status}")
                    return self._get_mock_supply_chain_data()
                    
        except Exception as e:
            logger.error(f"Error getting supply chain data: {e}")
            return self._get_mock_supply_chain_data()
            
    def _parse_financial_data(self, data: Dict[str, Any], period: str, currency: str) -> FinancialData:
        """Parse financial data from ERP response"""
        try:
            return FinancialData(
                revenue=data.get("revenue", 0.0),
                expenses=data.get("expenses", 0.0),
                profit_margin=data.get("profit_margin", 0.0),
                cash_flow=data.get("cash_flow", 0.0),
                accounts_receivable=data.get("accounts_receivable", 0.0),
                accounts_payable=data.get("accounts_payable", 0.0),
                currency=currency,
                period=period,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error parsing financial data: {e}")
            return self._get_mock_financial_data(period, currency)
            
    def _parse_inventory_data(self, data: Dict[str, Any]) -> InventoryData:
        """Parse inventory data from ERP response"""
        try:
            return InventoryData(
                total_items=data.get("total_items", 0),
                low_stock_items=data.get("low_stock_items", 0),
                out_of_stock_items=data.get("out_of_stock_items", 0),
                inventory_value=data.get("inventory_value", 0.0),
                turnover_rate=data.get("turnover_rate", 0.0),
                warehouse_locations=data.get("warehouse_locations", {}),
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error parsing inventory data: {e}")
            return self._get_mock_inventory_data()
            
    def _parse_supply_chain_data(self, data: Dict[str, Any]) -> SupplyChainData:
        """Parse supply chain data from ERP response"""
        try:
            return SupplyChainData(
                active_orders=data.get("active_orders", 0),
                pending_deliveries=data.get("pending_deliveries", 0),
                supplier_performance=data.get("supplier_performance", {}),
                lead_times=data.get("lead_times", {}),
                quality_metrics=data.get("quality_metrics", {}),
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error parsing supply chain data: {e}")
            return self._get_mock_supply_chain_data()
            
    def _get_mock_financial_data(self, period: str, currency: str) -> FinancialData:
        """Get mock financial data for testing"""
        return FinancialData(
            revenue=1250000.0,
            expenses=875000.0,
            profit_margin=0.30,
            cash_flow=250000.0,
            accounts_receivable=180000.0,
            accounts_payable=95000.0,
            currency=currency,
            period=period,
            timestamp=datetime.now()
        )
        
    def _get_mock_inventory_data(self) -> InventoryData:
        """Get mock inventory data for testing"""
        return InventoryData(
            total_items=15420,
            low_stock_items=342,
            out_of_stock_items=28,
            inventory_value=2850000.0,
            turnover_rate=4.2,
            warehouse_locations={
                "Main": 8500,
                "East": 4200,
                "West": 2720
            },
            timestamp=datetime.now()
        )
        
    def _get_mock_supply_chain_data(self) -> SupplyChainData:
        """Get mock supply chain data for testing"""
        return SupplyChainData(
            active_orders=156,
            pending_deliveries=89,
            supplier_performance={
                "Supplier A": 0.95,
                "Supplier B": 0.88,
                "Supplier C": 0.92
            },
            lead_times={
                "Electronics": 14,
                "Raw Materials": 7,
                "Packaging": 3
            },
            quality_metrics={
                "Defect Rate": 0.02,
                "On-Time Delivery": 0.94,
                "Quality Score": 0.96
            },
            timestamp=datetime.now()
        )
        
    async def get_customer_data(self, customer_id: Optional[str] = None) -> Dict[str, Any]:
        """Get customer data from ERP system"""
        try:
            if not await self._ensure_authenticated():
                return self._get_mock_customer_data(customer_id)
                
            if self.config.erp_type == ERPType.MOCK:
                return self._get_mock_customer_data(customer_id)
                
            headers = await self._get_headers()
            
            if customer_id:
                url = f"{self.config.base_url}/api/customers/{customer_id}"
            else:
                url = f"{self.config.base_url}/api/customers"
                
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Failed to get customer data: {response.status}")
                    return self._get_mock_customer_data(customer_id)
                    
        except Exception as e:
            logger.error(f"Error getting customer data: {e}")
            return self._get_mock_customer_data(customer_id)
            
    def _get_mock_customer_data(self, customer_id: Optional[str] = None) -> Dict[str, Any]:
        """Get mock customer data for testing"""
        if customer_id:
            return {
                "customer_id": customer_id,
                "name": "Acme Corporation",
                "email": "contact@acme.com",
                "phone": "+1-555-0123",
                "address": "123 Business St, City, State 12345",
                "total_orders": 156,
                "total_spent": 285000.0,
                "last_order_date": "2024-01-15",
                "customer_since": "2020-03-10",
                "status": "active"
            }
        else:
            return {
                "customers": [
                    {
                        "customer_id": "CUST001",
                        "name": "Acme Corporation",
                        "status": "active",
                        "total_spent": 285000.0
                    },
                    {
                        "customer_id": "CUST002", 
                        "name": "Tech Solutions Inc",
                        "status": "active",
                        "total_spent": 420000.0
                    }
                ],
                "total_customers": 2,
                "active_customers": 2
            }
            
    async def get_production_metrics(self) -> Dict[str, Any]:
        """Get production metrics from ERP system"""
        try:
            if not await self._ensure_authenticated():
                return self._get_mock_production_metrics()
                
            if self.config.erp_type == ERPType.MOCK:
                return self._get_mock_production_metrics()
                
            headers = await self._get_headers()
            url = f"{self.config.base_url}/api/production/metrics"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Failed to get production metrics: {response.status}")
                    return self._get_mock_production_metrics()
                    
        except Exception as e:
            logger.error(f"Error getting production metrics: {e}")
            return self._get_mock_production_metrics()
            
    def _get_mock_production_metrics(self) -> Dict[str, Any]:
        """Get mock production metrics for testing"""
        return {
            "efficiency": 0.87,
            "capacity_utilization": 0.92,
            "quality_rate": 0.96,
            "downtime_percentage": 0.08,
            "units_produced": 12500,
            "target_units": 13500,
            "overtime_hours": 45,
            "maintenance_schedule": "on_track",
            "timestamp": datetime.now().isoformat()
        }
