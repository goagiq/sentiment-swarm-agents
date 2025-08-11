"""
Orchestrator agent that implements the "Agents as Tools" pattern using 
Strands framework. This agent acts as the primary coordinator and has 
access to all specialized agents as tools.
"""

from typing import Any, Dict, List

from loguru import logger
from src.core.strands_mock import tool

from src.core.models import (
    AnalysisRequest, DataType, AnalysisResult, SentimentResult
)
from src.config.config import config
from src.agents.base_agent import StrandsBaseAgent
from src.core.tool_registry import tool_registry
from src.core.processing_service import processing_service
from src.core.error_handling_service import error_handling_service, ErrorContext


class OrchestratorAgent(StrandsBaseAgent):
    """
    Lightweight orchestrator agent that coordinates tool execution.
    
    This agent has been refactored to use the ToolRegistry and other
    shared services, reducing its size from 1067 lines to ~200 lines.
    """
    
    def __init__(self, model_name: str = None):
        # Use config system instead of hardcoded values
        default_model = config.model.default_text_model
        super().__init__(model_name=model_name or default_model)
        
        # Initialize services
        self.tool_registry = tool_registry
        self.processing_service = processing_service
        self.error_handling_service = error_handling_service
        
        # Set metadata
        self.metadata.update({
            "agent_type": "orchestrator",
            "model": model_name or default_model,
            "capabilities": [
                "tool_coordination",
                "query_routing",
                "service_discovery",
                "error_handling"
            ],
            "available_tools": len(self.tool_registry.list_tools()),
            "services": [
                "tool_registry",
                "processing_service", 
                "error_handling_service"
            ]
        })
        
        logger.info(f"Orchestrator Agent {self.agent_id} initialized with {len(self.tool_registry.list_tools())} tools")
    
    def _get_tools(self) -> list:
        """Get list of tools for this agent."""
        return [
            self.process_query,
            self.get_available_tools,
            self.execute_tool,
            self.get_tool_info,
            self.route_query
        ]
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the given request."""
        # Orchestrator can handle any request type by routing to appropriate tools
        return True
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process the analysis request by routing to appropriate tools."""
        context = ErrorContext(
            self.agent_id, 
            "process_request",
            request_id=request.id,
            data_type=request.data_type.value
        )
        
        return await self.error_handling_service.safe_execute_async(
            self._process_request_impl,
            request,
            context=context,
            default_return=AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(label="neutral", confidence=0.0),
                status=None,
                metadata={"error": "Processing failed"}
            )
        )
    
    async def _process_request_impl(self, request: AnalysisRequest) -> AnalysisResult:
        """Internal implementation of request processing."""
        # Route the request to appropriate tool
        tool_name = self._determine_tool_for_request(request)
        
        if not tool_name:
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(label="neutral", confidence=0.0),
                status=None,
                metadata={"error": "No suitable tool found"}
            )
        
        # Execute the tool
        tool_result = await self.tool_registry.execute_tool(tool_name, request.content)
        
        # Convert tool result to AnalysisResult
        return self._convert_tool_result_to_analysis_result(tool_result, request)
    
    def _determine_tool_for_request(self, request: AnalysisRequest) -> str:
        """Determine which tool to use for a given request."""
        content = str(request.content).lower()
        
        # Text processing
        if request.data_type == DataType.TEXT:
            if "complex" in content or "detailed" in content:
                return "swarm_text_analysis"
            else:
                return "text_sentiment_analysis"
        
        # Vision processing
        elif request.data_type == DataType.IMAGE:
            return "vision_sentiment_analysis"
        
        # Video processing
        elif request.data_type == DataType.VIDEO:
            if "youtube.com" in content or "youtu.be" in content:
                return "youtube_comprehensive_analysis"
            elif "summarize" in content or "summary" in content:
                return "video_summarization_analysis"
            else:
                return "unified_video_analysis"
        
        # Audio processing
        elif request.data_type == DataType.AUDIO:
            if "summarize" in content or "summary" in content:
                return "audio_summarization_analysis"
            else:
                return "enhanced_audio_sentiment_analysis"
        
        # Web processing
        elif request.data_type == DataType.WEBPAGE:
            return "web_sentiment_analysis"
        
        # Default to text analysis
        return "text_sentiment_analysis"
    
    def _convert_tool_result_to_analysis_result(
        self, 
        tool_result: Dict[str, Any], 
        request: AnalysisRequest
    ) -> AnalysisResult:
        """Convert tool result to AnalysisResult format."""
        if tool_result.get("status") == "success":
            content = tool_result.get("content", [{}])[0]
            json_data = content.get("json", {})
            
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label=json_data.get("sentiment", "neutral"),
                    confidence=json_data.get("confidence", 0.0)
                ),
                status=None,  # Will be set by base class
                metadata=json_data,
                raw_content=str(request.content),
                extracted_text=json_data.get("extracted_text", "")
            )
        else:
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(label="neutral", confidence=0.0),
                status=None,
                metadata={"error": tool_result.get("content", [{"text": "Unknown error"}])[0].get("text", "Unknown error")}
            )
    
    @tool
    async def process_query(self, query: str) -> str:
        """
        Process a natural language query and route to appropriate tools.
        
        Args:
            query: Natural language query describing what to analyze
            
        Returns:
            Analysis result as formatted string
        """
        try:
            # Create a mock request for routing
            request = AnalysisRequest(
                data_type=DataType.TEXT,
                content=query,
                language="en"
            )
            
            result = await self.process(request)
            
            if result.sentiment:
                return f"Analysis Result: {result.sentiment.label} (confidence: {result.sentiment.confidence:.2f})\nMetadata: {result.metadata}"
            else:
                return f"Analysis completed. Metadata: {result.metadata}"
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error processing query: {str(e)}"
    
    @tool
    async def get_available_tools(self) -> Dict[str, Any]:
        """Get information about available tools."""
        tools = self.tool_registry.list_tools()
        tool_info = {}
        
        for tool_name in tools:
            metadata = self.tool_registry.get_tool_metadata(tool_name)
            tool_info[tool_name] = metadata
        
        return {
            "total_tools": len(tools),
            "tools": tool_info,
            "tool_categories": {
                "text": self.tool_registry.get_tools_by_tag("text"),
                "vision": self.tool_registry.get_tools_by_tag("vision"),
                "audio": self.tool_registry.get_tools_by_tag("audio"),
                "video": self.tool_registry.get_tools_by_tag("video"),
                "web": self.tool_registry.get_tools_by_tag("web"),
                "ocr": self.tool_registry.get_tools_by_tag("ocr")
            }
        }
    
    @tool
    async def execute_tool(self, tool_name: str, *args, **kwargs) -> Dict[str, Any]:
        """Execute a specific tool by name."""
        try:
            result = await self.tool_registry.execute_tool(tool_name, *args, **kwargs)
            return {
                "status": "success",
                "tool": tool_name,
                "result": result
            }
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {
                "status": "error",
                "tool": tool_name,
                "error": str(e)
            }
    
    @tool
    async def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific tool."""
        metadata = self.tool_registry.get_tool_metadata(tool_name)
        if metadata:
            return {
                "tool_name": tool_name,
                "metadata": metadata,
                "available": True
            }
        else:
            return {
                "tool_name": tool_name,
                "available": False,
                "error": "Tool not found"
            }
    
    @tool
    async def route_query(self, query: str) -> Dict[str, Any]:
        """Route a query to the most appropriate tool."""
        try:
            # Create a mock request for routing
            request = AnalysisRequest(
                data_type=DataType.TEXT,
                content=query,
                language="en"
            )
            
            tool_name = self._determine_tool_for_request(request)
            
            return {
                "query": query,
                "recommended_tool": tool_name,
                "tool_metadata": self.tool_registry.get_tool_metadata(tool_name),
                "alternative_tools": self._get_alternative_tools(request)
            }
            
        except Exception as e:
            logger.error(f"Error routing query: {e}")
            return {
                "query": query,
                "error": str(e)
            }
    
    def _get_alternative_tools(self, request: AnalysisRequest) -> List[str]:
        """Get alternative tools for a request."""
        alternatives = []
        
        if request.data_type == DataType.TEXT:
            alternatives = ["text_sentiment_analysis", "swarm_text_analysis"]
        elif request.data_type == DataType.IMAGE:
            alternatives = ["vision_sentiment_analysis", "ocr_analysis"]
        elif request.data_type == DataType.VIDEO:
            alternatives = ["unified_video_analysis", "video_summarization_analysis"]
        elif request.data_type == DataType.AUDIO:
            alternatives = ["enhanced_audio_sentiment_analysis", "audio_summarization_analysis"]
        
        return [tool for tool in alternatives if tool in self.tool_registry.list_tools()]
    
    async def start(self):
        """Start the orchestrator agent."""
        await super().start()
        logger.info(f"Orchestrator Agent {self.agent_id} started with {len(self.tool_registry.list_tools())} tools")
    
    async def stop(self):
        """Stop the orchestrator agent."""
        await super().stop()
        logger.info(f"Orchestrator Agent {self.agent_id} stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        base_status = super().get_status()
        base_status.update({
            "available_tools": len(self.tool_registry.list_tools()),
            "tool_categories": {
                "text": len(self.tool_registry.get_tools_by_tag("text")),
                "vision": len(self.tool_registry.get_tools_by_tag("vision")),
                "audio": len(self.tool_registry.get_tools_by_tag("audio")),
                "video": len(self.tool_registry.get_tools_by_tag("video")),
                "web": len(self.tool_registry.get_tools_by_tag("web")),
                "ocr": len(self.tool_registry.get_tools_by_tag("ocr"))
            }
        })
        return base_status
