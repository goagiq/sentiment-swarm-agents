"""
External Data Agent for integrating with databases and external APIs.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from src.agents.base_agent import StrandsBaseAgent
from src.core.models import AnalysisRequest, AnalysisResult, DataType, SentimentResult
from src.core.error_handler import with_error_handling


class DatabaseManager:
    """Manage database connections and queries."""
    
    def __init__(self):
        self.connections = {}
        self.cache = {}
    
    @with_error_handling("database_connection")
    async def connect_database(
        self,
        database_type: str,
        connection_string: str,
        query: str,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Connect to database and execute query."""
        try:
            logger.info(f"Connecting to {database_type} database")
            
            # Mock database connection and query execution
            # In production, this would use actual database drivers
            if database_type == "mongodb":
                result = await self._query_mongodb(connection_string, query)
            elif database_type == "postgresql":
                result = await self._query_postgresql(connection_string, query)
            elif database_type == "mysql":
                result = await self._query_mysql(connection_string, query)
            elif database_type == "elasticsearch":
                result = await self._query_elasticsearch(connection_string, query)
            else:
                return {"error": f"Database type {database_type} not supported"}
            
            # Add metadata if requested
            if include_metadata:
                result["metadata"] = {
                    "database_type": database_type,
                    "query_executed": query,
                    "timestamp": datetime.now().isoformat(),
                    "connection_string": connection_string[:20] + "..." if len(connection_string) > 20 else connection_string
                }
            
            logger.info(f"Database query executed successfully for {database_type}")
            return result
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return {"error": str(e)}
    
    async def _query_mongodb(self, connection_string: str, query: str) -> Dict[str, Any]:
        """Query MongoDB database."""
        # Mock MongoDB query result
        return {
            "database_type": "mongodb",
            "query": query,
            "results": [
                {"id": "1", "content": "Sample document 1", "timestamp": "2024-01-01"},
                {"id": "2", "content": "Sample document 2", "timestamp": "2024-01-02"}
            ],
            "total_documents": 2,
            "execution_time": 0.15
        }
    
    async def _query_postgresql(self, connection_string: str, query: str) -> Dict[str, Any]:
        """Query PostgreSQL database."""
        # Mock PostgreSQL query result
        return {
            "database_type": "postgresql",
            "query": query,
            "results": [
                {"id": 1, "content": "Sample row 1", "created_at": "2024-01-01"},
                {"id": 2, "content": "Sample row 2", "created_at": "2024-01-02"}
            ],
            "total_rows": 2,
            "execution_time": 0.12
        }
    
    async def _query_mysql(self, connection_string: str, query: str) -> Dict[str, Any]:
        """Query MySQL database."""
        # Mock MySQL query result
        return {
            "database_type": "mysql",
            "query": query,
            "results": [
                {"id": 1, "content": "Sample record 1", "created_at": "2024-01-01"},
                {"id": 2, "content": "Sample record 2", "created_at": "2024-01-02"}
            ],
            "total_records": 2,
            "execution_time": 0.10
        }
    
    async def _query_elasticsearch(self, connection_string: str, query: str) -> Dict[str, Any]:
        """Query Elasticsearch database."""
        # Mock Elasticsearch query result
        return {
            "database_type": "elasticsearch",
            "query": query,
            "results": [
                {"_id": "1", "_source": {"content": "Sample document 1"}},
                {"_id": "2", "_source": {"content": "Sample document 2"}}
            ],
            "total_hits": 2,
            "execution_time": 0.08
        }


class APIManager:
    """Manage external API integrations."""
    
    def __init__(self):
        self.cache = {}
        self.rate_limits = {}
    
    @with_error_handling("api_integration")
    async def fetch_api_data(
        self,
        api_endpoint: str,
        api_type: str = "rest",
        parameters: Dict[str, Any] = {},
        authentication: Dict[str, str] = {},
        include_caching: bool = True
    ) -> Dict[str, Any]:
        """Fetch data from external API."""
        try:
            logger.info(f"Fetching data from API: {api_endpoint}")
            
            # Check cache first
            cache_key = f"{api_endpoint}_{hash(str(parameters))}"
            if include_caching and cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if datetime.now() - cached_data["timestamp"] < timedelta(minutes=30):
                    logger.info("Returning cached API data")
                    return cached_data["data"]
            
            # Mock API call based on type
            if api_type == "rest":
                result = await self._fetch_rest_api(api_endpoint, parameters, authentication)
            elif api_type == "graphql":
                result = await self._fetch_graphql_api(api_endpoint, parameters, authentication)
            elif api_type == "soap":
                result = await self._fetch_soap_api(api_endpoint, parameters, authentication)
            else:
                return {"error": f"API type {api_type} not supported"}
            
            # Cache result if requested
            if include_caching:
                self.cache[cache_key] = {
                    "data": result,
                    "timestamp": datetime.now()
                }
            
            logger.info(f"API data fetched successfully from {api_endpoint}")
            return result
            
        except Exception as e:
            logger.error(f"API fetch failed: {e}")
            return {"error": str(e)}
    
    async def _fetch_rest_api(
        self,
        api_endpoint: str,
        parameters: Dict[str, Any],
        authentication: Dict[str, str]
    ) -> Dict[str, Any]:
        """Fetch data from REST API."""
        # Mock REST API response
        return {
            "api_type": "rest",
            "endpoint": api_endpoint,
            "parameters": parameters,
            "data": [
                {"id": 1, "title": "Sample API data 1", "content": "Content 1"},
                {"id": 2, "title": "Sample API data 2", "content": "Content 2"}
            ],
            "total_items": 2,
            "response_time": 0.25
        }
    
    async def _fetch_graphql_api(
        self,
        api_endpoint: str,
        parameters: Dict[str, Any],
        authentication: Dict[str, str]
    ) -> Dict[str, Any]:
        """Fetch data from GraphQL API."""
        # Mock GraphQL API response
        return {
            "api_type": "graphql",
            "endpoint": api_endpoint,
            "query": parameters.get("query", ""),
            "data": {
                "items": [
                    {"id": "1", "title": "GraphQL item 1"},
                    {"id": "2", "title": "GraphQL item 2"}
                ]
            },
            "response_time": 0.20
        }
    
    async def _fetch_soap_api(
        self,
        api_endpoint: str,
        parameters: Dict[str, Any],
        authentication: Dict[str, str]
    ) -> Dict[str, Any]:
        """Fetch data from SOAP API."""
        # Mock SOAP API response
        return {
            "api_type": "soap",
            "endpoint": api_endpoint,
            "operation": parameters.get("operation", ""),
            "data": [
                {"id": 1, "name": "SOAP item 1"},
                {"id": 2, "name": "SOAP item 2"}
            ],
            "response_time": 0.30
        }


class DataSourceManager:
    """Manage external data sources."""
    
    def __init__(self):
        self.sources = {}
        self.validators = {}
    
    @with_error_handling("data_source_management")
    async def manage_data_sources(
        self,
        action: str,
        source_config: Dict[str, Any] = {},
        include_validation: bool = True
    ) -> Dict[str, Any]:
        """Manage external data sources."""
        try:
            logger.info(f"Managing data sources with action: {action}")
            
            if action == "add":
                result = await self._add_data_source(source_config, include_validation)
            elif action == "update":
                result = await self._update_data_source(source_config, include_validation)
            elif action == "remove":
                result = await self._remove_data_source(source_config)
            elif action == "list":
                result = await self._list_data_sources()
            elif action == "test":
                result = await self._test_data_source(source_config)
            else:
                return {"error": f"Action {action} not supported"}
            
            logger.info(f"Data source management completed: {action}")
            return result
            
        except Exception as e:
            logger.error(f"Data source management failed: {e}")
            return {"error": str(e)}
    
    async def _add_data_source(self, source_config: Dict[str, Any], include_validation: bool) -> Dict[str, Any]:
        """Add a new data source."""
        source_id = source_config.get("id", f"source_{len(self.sources) + 1}")
        
        if include_validation:
            validation_result = await self._validate_source_config(source_config)
            if not validation_result.get("valid", False):
                return {"error": f"Invalid source configuration: {validation_result.get('errors', [])}"}
        
        self.sources[source_id] = {
            **source_config,
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        return {
            "action": "add",
            "source_id": source_id,
            "status": "success",
            "message": f"Data source {source_id} added successfully"
        }
    
    async def _update_data_source(self, source_config: Dict[str, Any], include_validation: bool) -> Dict[str, Any]:
        """Update an existing data source."""
        source_id = source_config.get("id")
        if not source_id or source_id not in self.sources:
            return {"error": f"Data source {source_id} not found"}
        
        if include_validation:
            validation_result = await self._validate_source_config(source_config)
            if not validation_result.get("valid", False):
                return {"error": f"Invalid source configuration: {validation_result.get('errors', [])}"}
        
        self.sources[source_id].update({
            **source_config,
            "updated_at": datetime.now().isoformat()
        })
        
        return {
            "action": "update",
            "source_id": source_id,
            "status": "success",
            "message": f"Data source {source_id} updated successfully"
        }
    
    async def _remove_data_source(self, source_config: Dict[str, Any]) -> Dict[str, Any]:
        """Remove a data source."""
        source_id = source_config.get("id")
        if not source_id or source_id not in self.sources:
            return {"error": f"Data source {source_id} not found"}
        
        del self.sources[source_id]
        
        return {
            "action": "remove",
            "source_id": source_id,
            "status": "success",
            "message": f"Data source {source_id} removed successfully"
        }
    
    async def _list_data_sources(self) -> Dict[str, Any]:
        """List all data sources."""
        return {
            "action": "list",
            "total_sources": len(self.sources),
            "sources": list(self.sources.keys()),
            "source_details": self.sources
        }
    
    async def _test_data_source(self, source_config: Dict[str, Any]) -> Dict[str, Any]:
        """Test a data source connection."""
        source_type = source_config.get("type", "unknown")
        
        # Mock connection test
        test_result = {
            "source_id": source_config.get("id", "test_source"),
            "source_type": source_type,
            "connection_status": "success",
            "response_time": 0.15,
            "test_timestamp": datetime.now().isoformat()
        }
        
        return {
            "action": "test",
            "test_result": test_result
        }
    
    async def _validate_source_config(self, source_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data source configuration."""
        errors = []
        
        required_fields = ["id", "type", "connection_string"]
        for field in required_fields:
            if field not in source_config:
                errors.append(f"Missing required field: {field}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }


class ExternalDataAgent(StrandsBaseAgent):
    """
    External Data Agent for integrating with databases and external APIs.
    
    Supports:
    - Database connections and queries
    - External API integrations
    - Data source management
    - Connection pooling and caching
    """
    
    def __init__(self, model_name: Optional[str] = None, **kwargs):
        super().__init__(model_name=model_name or "mistral-small3.1:latest", **kwargs)
        
        # Initialize external data components
        self.database_manager = DatabaseManager()
        self.api_manager = APIManager()
        self.data_source_manager = DataSourceManager()
        
        # Set metadata
        self.metadata["agent_type"] = "external_data"
        self.metadata["capabilities"] = [
            "database_integration",
            "api_integration",
            "data_source_management",
            "connection_pooling",
            "data_caching"
        ]
        self.metadata["supported_databases"] = ["mongodb", "postgresql", "mysql", "elasticsearch"]
        self.metadata["supported_apis"] = ["rest", "graphql", "soap"]
        
        logger.info("ExternalDataAgent initialized successfully")
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the given request."""
        # External data agent can process database and API related requests
        return request.data_type in [DataType.TEXT, DataType.DATABASE, DataType.API, DataType.GENERAL]
    
    @with_error_handling("external_data_processing")
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process external data requests."""
        try:
            logger.info(f"Processing external data request: {request.data_type}")
            
            start_time = datetime.now()
            
            # Route request based on data type and metadata
            if request.data_type == DataType.DATABASE:
                result = await self._process_database_request(request)
            elif request.data_type == DataType.API:
                result = await self._process_api_request(request)
            elif request.data_type == DataType.TEXT:
                result = await self._process_text_request(request)
            else:
                result = await self._process_general_request(request)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                status="completed",
                sentiment=result.get("sentiment", SentimentResult(label="neutral", confidence=0.5, reasoning="External data analysis completed")),
                extracted_text=result.get("extracted_text", ""),
                metadata=result.get("metadata", {}),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"External data processing failed: {e}")
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                status="failed",
                sentiment=SentimentResult(label="neutral", confidence=0.0, reasoning=f"Processing failed: {str(e)}"),
                metadata={"error": str(e)},
                processing_time=0.0
            )
    
    async def _process_database_request(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Process database related requests."""
        # Extract parameters from request metadata
        database_type = request.metadata.get("database_type", "mongodb")
        connection_string = request.metadata.get("connection_string", "")
        query = request.metadata.get("query", "")
        include_metadata = request.metadata.get("include_metadata", True)
        
        return await self.database_manager.connect_database(
            database_type, connection_string, query, include_metadata
        )
    
    async def _process_api_request(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Process API related requests."""
        # Extract parameters from request metadata
        api_endpoint = request.metadata.get("api_endpoint", "")
        api_type = request.metadata.get("api_type", "rest")
        parameters = request.metadata.get("parameters", {})
        authentication = request.metadata.get("authentication", {})
        include_caching = request.metadata.get("include_caching", True)
        
        return await self.api_manager.fetch_api_data(
            api_endpoint, api_type, parameters, authentication, include_caching
        )
    
    async def _process_text_request(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Process text-based external data requests."""
        content = request.content
        
        # Analyze external data content
        analysis = {
            "content_type": "external_data_text",
            "data_source": request.metadata.get("data_source", "unknown"),
            "content_length": len(content),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return analysis
    
    async def _process_general_request(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Process general external data requests."""
        return await self._process_text_request(request)
    
    async def connect_database_source(
        self,
        database_type: str,
        connection_string: str,
        query: str,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Connect and query database sources."""
        return await self.database_manager.connect_database(
            database_type, connection_string, query, include_metadata
        )
    
    async def fetch_external_api_data(
        self,
        api_endpoint: str,
        api_type: str = "rest",
        parameters: Dict[str, Any] = {},
        authentication: Dict[str, str] = {},
        include_caching: bool = True
    ) -> Dict[str, Any]:
        """Fetch data from external APIs."""
        return await self.api_manager.fetch_api_data(
            api_endpoint, api_type, parameters, authentication, include_caching
        )
    
    async def manage_data_sources(
        self,
        action: str,
        source_config: Dict[str, Any] = {},
        include_validation: bool = True
    ) -> Dict[str, Any]:
        """Manage external data sources."""
        return await self.data_source_manager.manage_data_sources(
            action, source_config, include_validation
        )
