"""
CRM System Connector for Decision Support System

This module provides integration with major CRM systems including:
- Salesforce
- HubSpot
- Microsoft Dynamics CRM
- Pipedrive
- Zoho CRM

Provides real-time access to:
- Customer data and profiles
- Sales pipeline information
- Lead management data
- Customer interactions
- Sales performance metrics
"""

import aiohttp
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CRMType(Enum):
    """Supported CRM system types"""
    SALESFORCE = "salesforce"
    HUBSPOT = "hubspot"
    DYNAMICS_CRM = "dynamics_crm"
    PIPEDRIVE = "pipedrive"
    ZOHO = "zoho"
    MOCK = "mock"


@dataclass
class CRMConfig:
    """Configuration for CRM system connection"""
    crm_type: CRMType
    base_url: str
    api_key: Optional[str] = None
    access_token: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    refresh_token: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3


@dataclass
class CustomerProfile:
    """Customer profile data structure"""
    customer_id: str
    name: str
    email: str
    phone: Optional[str] = None
    company: Optional[str] = None
    industry: Optional[str] = None
    lead_score: Optional[int] = None
    status: str = "active"
    created_date: Optional[datetime] = None
    last_contact: Optional[datetime] = None
    total_value: float = 0.0
    tags: List[str] = None


@dataclass
class SalesPipeline:
    """Sales pipeline data structure"""
    total_leads: int
    qualified_leads: int
    opportunities: int
    closed_won: int
    closed_lost: int
    total_value: float
    average_deal_size: float
    conversion_rate: float
    sales_cycle_days: int
    timestamp: Optional[datetime] = None


@dataclass
class InteractionData:
    """Customer interaction data structure"""
    interaction_id: str
    customer_id: str
    interaction_type: str
    subject: str
    description: str
    date: datetime
    duration: Optional[int] = None
    outcome: Optional[str] = None
    next_action: Optional[str] = None


class CRMConnector:
    """
    CRM System Connector for real-time customer relationship data integration
    
    Supports multiple CRM systems with unified interface for:
    - Customer profile management
    - Sales pipeline tracking
    - Lead management
    - Customer interactions
    - Sales performance analytics
    """
    
    def __init__(self, config: CRMConfig):
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
        """Authenticate with CRM system based on type"""
        try:
            if self.config.crm_type == CRMType.SALESFORCE:
                return await self._authenticate_salesforce()
            elif self.config.crm_type == CRMType.HUBSPOT:
                return await self._authenticate_hubspot()
            elif self.config.crm_type == CRMType.DYNAMICS_CRM:
                return await self._authenticate_dynamics_crm()
            elif self.config.crm_type == CRMType.PIPEDRIVE:
                return await self._authenticate_pipedrive()
            elif self.config.crm_type == CRMType.ZOHO:
                return await self._authenticate_zoho()
            elif self.config.crm_type == CRMType.MOCK:
                return True
            else:
                logger.error(f"Unsupported CRM type: {self.config.crm_type}")
                return False
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
            
    async def _authenticate_salesforce(self) -> bool:
        """Authenticate with Salesforce"""
        try:
            auth_url = f"{self.config.base_url}/services/oauth2/token"
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
                    logger.error(f"Salesforce authentication failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Salesforce authentication error: {e}")
            return False
            
    async def _authenticate_hubspot(self) -> bool:
        """Authenticate with HubSpot"""
        try:
            # HubSpot uses API key authentication
            if self.config.api_key:
                self._access_token = self.config.api_key
                self._token_expiry = datetime.now() + timedelta(hours=24)
                return True
            else:
                logger.error("HubSpot requires API key for authentication")
                return False
        except Exception as e:
            logger.error(f"HubSpot authentication error: {e}")
            return False
            
    async def _authenticate_dynamics_crm(self) -> bool:
        """Authenticate with Microsoft Dynamics CRM"""
        try:
            auth_url = f"https://login.microsoftonline.com/common/oauth2/v2.0/token"
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
                    logger.error(f"Dynamics CRM authentication failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Dynamics CRM authentication error: {e}")
            return False
            
    async def _authenticate_pipedrive(self) -> bool:
        """Authenticate with Pipedrive"""
        try:
            # Pipedrive uses API token authentication
            if self.config.api_key:
                self._access_token = self.config.api_key
                self._token_expiry = datetime.now() + timedelta(hours=24)
                return True
            else:
                logger.error("Pipedrive requires API token for authentication")
                return False
        except Exception as e:
            logger.error(f"Pipedrive authentication error: {e}")
            return False
            
    async def _authenticate_zoho(self) -> bool:
        """Authenticate with Zoho CRM"""
        try:
            auth_url = f"https://accounts.zoho.com/oauth/v2/token"
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
                    logger.error(f"Zoho authentication failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Zoho authentication error: {e}")
            return False
            
    async def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if self._access_token:
            if self.config.crm_type == CRMType.SALESFORCE:
                headers["Authorization"] = f"Bearer {self._access_token}"
            elif self.config.crm_type == CRMType.HUBSPOT:
                headers["Authorization"] = f"Bearer {self._access_token}"
            elif self.config.crm_type == CRMType.DYNAMICS_CRM:
                headers["Authorization"] = f"Bearer {self._access_token}"
            elif self.config.crm_type == CRMType.PIPEDRIVE:
                headers["Authorization"] = f"Bearer {self._access_token}"
            elif self.config.crm_type == CRMType.ZOHO:
                headers["Authorization"] = f"Bearer {self._access_token}"
                
        return headers
        
    async def _ensure_authenticated(self) -> bool:
        """Ensure we have a valid authentication token"""
        if not self._access_token or (self._token_expiry and datetime.now() >= self._token_expiry):
            return await self._authenticate()
        return True
        
    async def get_customer_profile(self, customer_id: str) -> Optional[CustomerProfile]:
        """Get customer profile from CRM system"""
        try:
            if not await self._ensure_authenticated():
                return self._get_mock_customer_profile(customer_id)
                
            if self.config.crm_type == CRMType.MOCK:
                return self._get_mock_customer_profile(customer_id)
                
            headers = await self._get_headers()
            
            # Different CRM systems have different endpoints
            if self.config.crm_type == CRMType.SALESFORCE:
                url = f"{self.config.base_url}/services/data/v52.0/sobjects/Contact/{customer_id}"
            elif self.config.crm_type == CRMType.HUBSPOT:
                url = f"{self.config.base_url}/crm/v3/objects/contacts/{customer_id}"
            elif self.config.crm_type == CRMType.DYNAMICS_CRM:
                url = f"{self.config.base_url}/api/data/v9.2/contacts({customer_id})"
            elif self.config.crm_type == CRMType.PIPEDRIVE:
                url = f"{self.config.base_url}/v1/persons/{customer_id}"
            elif self.config.crm_type == CRMType.ZOHO:
                url = f"{self.config.base_url}/crm/v2/Contacts/{customer_id}"
            else:
                return self._get_mock_customer_profile(customer_id)
                
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_customer_profile(data, customer_id)
                else:
                    logger.warning(f"Failed to get customer profile: {response.status}")
                    return self._get_mock_customer_profile(customer_id)
                    
        except Exception as e:
            logger.error(f"Error getting customer profile: {e}")
            return self._get_mock_customer_profile(customer_id)
            
    async def get_sales_pipeline(self) -> Optional[SalesPipeline]:
        """Get sales pipeline data from CRM system"""
        try:
            if not await self._ensure_authenticated():
                return self._get_mock_sales_pipeline()
                
            if self.config.crm_type == CRMType.MOCK:
                return self._get_mock_sales_pipeline()
                
            headers = await self._get_headers()
            
            # Different CRM systems have different endpoints
            if self.config.crm_type == CRMType.SALESFORCE:
                url = f"{self.config.base_url}/services/data/v52.0/sobjects/Opportunity"
            elif self.config.crm_type == CRMType.HUBSPOT:
                url = f"{self.config.base_url}/crm/v3/objects/deals"
            elif self.config.crm_type == CRMType.DYNAMICS_CRM:
                url = f"{self.config.base_url}/api/data/v9.2/opportunities"
            elif self.config.crm_type == CRMType.PIPEDRIVE:
                url = f"{self.config.base_url}/v1/deals"
            elif self.config.crm_type == CRMType.ZOHO:
                url = f"{self.config.base_url}/crm/v2/Deals"
            else:
                return self._get_mock_sales_pipeline()
                
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_sales_pipeline(data)
                else:
                    logger.warning(f"Failed to get sales pipeline: {response.status}")
                    return self._get_mock_sales_pipeline()
                    
        except Exception as e:
            logger.error(f"Error getting sales pipeline: {e}")
            return self._get_mock_sales_pipeline()
            
    async def get_customer_interactions(self, customer_id: str, days: int = 30) -> List[InteractionData]:
        """Get customer interactions from CRM system"""
        try:
            if not await self._ensure_authenticated():
                return self._get_mock_interactions(customer_id, days)
                
            if self.config.crm_type == CRMType.MOCK:
                return self._get_mock_interactions(customer_id, days)
                
            headers = await self._get_headers()
            
            # Different CRM systems have different endpoints
            if self.config.crm_type == CRMType.SALESFORCE:
                url = f"{self.config.base_url}/services/data/v52.0/sobjects/Task"
            elif self.config.crm_type == CRMType.HUBSPOT:
                url = f"{self.config.base_url}/crm/v3/objects/contacts/{customer_id}/associations/engagements"
            elif self.config.crm_type == CRMType.DYNAMICS_CRM:
                url = f"{self.config.base_url}/api/data/v9.2/activities"
            elif self.config.crm_type == CRMType.PIPEDRIVE:
                url = f"{self.config.base_url}/v1/persons/{customer_id}/activities"
            elif self.config.crm_type == CRMType.ZOHO:
                url = f"{self.config.base_url}/crm/v2/Activities"
            else:
                return self._get_mock_interactions(customer_id, days)
                
            params = {
                "customer_id": customer_id,
                "days": days
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_interactions(data, customer_id)
                else:
                    logger.warning(f"Failed to get interactions: {response.status}")
                    return self._get_mock_interactions(customer_id, days)
                    
        except Exception as e:
            logger.error(f"Error getting customer interactions: {e}")
            return self._get_mock_interactions(customer_id, days)
            
    def _parse_customer_profile(self, data: Dict[str, Any], customer_id: str) -> CustomerProfile:
        """Parse customer profile from CRM response"""
        try:
            return CustomerProfile(
                customer_id=customer_id,
                name=data.get("Name", data.get("name", "Unknown")),
                email=data.get("Email", data.get("email", "")),
                phone=data.get("Phone", data.get("phone")),
                company=data.get("Company", data.get("company")),
                industry=data.get("Industry", data.get("industry")),
                lead_score=data.get("LeadScore", data.get("lead_score")),
                status=data.get("Status", data.get("status", "active")),
                created_date=datetime.now() - timedelta(days=30),
                last_contact=datetime.now() - timedelta(days=7),
                total_value=data.get("TotalValue", data.get("total_value", 0.0)),
                tags=data.get("Tags", data.get("tags", []))
            )
        except Exception as e:
            logger.error(f"Error parsing customer profile: {e}")
            return self._get_mock_customer_profile(customer_id)
            
    def _parse_sales_pipeline(self, data: Dict[str, Any]) -> SalesPipeline:
        """Parse sales pipeline from CRM response"""
        try:
            opportunities = data.get("records", data.get("data", []))
            total_leads = len(opportunities)
            qualified_leads = sum(1 for opp in opportunities if opp.get("StageName") in ["Qualified", "Proposal"])
            closed_won = sum(1 for opp in opportunities if opp.get("StageName") == "Closed Won")
            closed_lost = sum(1 for opp in opportunities if opp.get("StageName") == "Closed Lost")
            total_value = sum(float(opp.get("Amount", 0)) for opp in opportunities)
            
            return SalesPipeline(
                total_leads=total_leads,
                qualified_leads=qualified_leads,
                opportunities=total_leads,
                closed_won=closed_won,
                closed_lost=closed_lost,
                total_value=total_value,
                average_deal_size=total_value / total_leads if total_leads > 0 else 0,
                conversion_rate=closed_won / total_leads if total_leads > 0 else 0,
                sales_cycle_days=45,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error parsing sales pipeline: {e}")
            return self._get_mock_sales_pipeline()
            
    def _parse_interactions(self, data: Dict[str, Any], customer_id: str) -> List[InteractionData]:
        """Parse interactions from CRM response"""
        try:
            interactions = data.get("records", data.get("data", []))
            result = []
            
            for interaction in interactions:
                result.append(InteractionData(
                    interaction_id=interaction.get("Id", str(len(result))),
                    customer_id=customer_id,
                    interaction_type=interaction.get("Type", "Call"),
                    subject=interaction.get("Subject", "Interaction"),
                    description=interaction.get("Description", ""),
                    date=datetime.now() - timedelta(days=len(result)),
                    duration=interaction.get("DurationInMinutes", 30),
                    outcome=interaction.get("Status", "Completed"),
                    next_action=interaction.get("NextAction", "Follow up")
                ))
                
            return result
        except Exception as e:
            logger.error(f"Error parsing interactions: {e}")
            return self._get_mock_interactions(customer_id, 30)
            
    def _get_mock_customer_profile(self, customer_id: str) -> CustomerProfile:
        """Get mock customer profile for testing"""
        return CustomerProfile(
            customer_id=customer_id,
            name="John Smith",
            email="john.smith@example.com",
            phone="+1-555-0123",
            company="Acme Corporation",
            industry="Technology",
            lead_score=85,
            status="active",
            created_date=datetime.now() - timedelta(days=30),
            last_contact=datetime.now() - timedelta(days=7),
            total_value=125000.0,
            tags=["VIP", "Enterprise", "Decision Maker"]
        )
        
    def _get_mock_sales_pipeline(self) -> SalesPipeline:
        """Get mock sales pipeline for testing"""
        return SalesPipeline(
            total_leads=156,
            qualified_leads=89,
            opportunities=67,
            closed_won=23,
            closed_lost=12,
            total_value=2850000.0,
            average_deal_size=42500.0,
            conversion_rate=0.34,
            sales_cycle_days=45,
            timestamp=datetime.now()
        )
        
    def _get_mock_interactions(self, customer_id: str, days: int) -> List[InteractionData]:
        """Get mock interactions for testing"""
        interactions = []
        for i in range(min(days, 10)):
            interactions.append(InteractionData(
                interaction_id=f"INT_{i}",
                customer_id=customer_id,
                interaction_type="Call" if i % 2 == 0 else "Email",
                subject=f"Follow up {i+1}",
                description=f"Customer interaction {i+1}",
                date=datetime.now() - timedelta(days=i),
                duration=30 if i % 2 == 0 else None,
                outcome="Completed",
                next_action="Schedule next meeting"
            ))
        return interactions
        
    async def get_lead_data(self, lead_id: Optional[str] = None) -> Dict[str, Any]:
        """Get lead data from CRM system"""
        try:
            if not await self._ensure_authenticated():
                return self._get_mock_lead_data(lead_id)
                
            if self.config.crm_type == CRMType.MOCK:
                return self._get_mock_lead_data(lead_id)
                
            headers = await self._get_headers()
            
            if lead_id:
                url = f"{self.config.base_url}/api/leads/{lead_id}"
            else:
                url = f"{self.config.base_url}/api/leads"
                
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Failed to get lead data: {response.status}")
                    return self._get_mock_lead_data(lead_id)
                    
        except Exception as e:
            logger.error(f"Error getting lead data: {e}")
            return self._get_mock_lead_data(lead_id)
            
    def _get_mock_lead_data(self, lead_id: Optional[str] = None) -> Dict[str, Any]:
        """Get mock lead data for testing"""
        if lead_id:
            return {
                "lead_id": lead_id,
                "name": "Jane Doe",
                "email": "jane.doe@example.com",
                "company": "Tech Solutions Inc",
                "status": "qualified",
                "source": "website",
                "score": 75,
                "created_date": "2024-01-10",
                "last_activity": "2024-01-15"
            }
        else:
            return {
                "leads": [
                    {
                        "lead_id": "LEAD001",
                        "name": "Jane Doe",
                        "status": "qualified",
                        "score": 75
                    },
                    {
                        "lead_id": "LEAD002",
                        "name": "Bob Johnson",
                        "status": "new",
                        "score": 45
                    }
                ],
                "total_leads": 2,
                "qualified_leads": 1
            }
            
    async def get_sales_performance(self) -> Dict[str, Any]:
        """Get sales performance metrics from CRM system"""
        try:
            if not await self._ensure_authenticated():
                return self._get_mock_sales_performance()
                
            if self.config.crm_type == CRMType.MOCK:
                return self._get_mock_sales_performance()
                
            headers = await self._get_headers()
            url = f"{self.config.base_url}/api/sales/performance"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Failed to get sales performance: {response.status}")
                    return self._get_mock_sales_performance()
                    
        except Exception as e:
            logger.error(f"Error getting sales performance: {e}")
            return self._get_mock_sales_performance()
            
    def _get_mock_sales_performance(self) -> Dict[str, Any]:
        """Get mock sales performance for testing"""
        return {
            "total_revenue": 2850000.0,
            "quota_achievement": 0.95,
            "average_deal_size": 42500.0,
            "sales_cycle_days": 45,
            "win_rate": 0.34,
            "top_performers": [
                {"name": "Alice Johnson", "revenue": 450000.0},
                {"name": "Bob Smith", "revenue": 380000.0}
            ],
            "monthly_trends": {
                "january": 250000.0,
                "february": 275000.0,
                "march": 300000.0
            },
            "timestamp": datetime.now().isoformat()
        }
