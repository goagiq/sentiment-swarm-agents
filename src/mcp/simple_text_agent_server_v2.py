"""
SimpleTextAgent MCP Server v2 - Using mcp.server.Server for MCP version 2 support.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from loguru import logger

from mcp.server import Server
from mcp.server.models import (
    InitializationOptions, ServerCapabilities, Tool, ToolCallRequest, ToolCallResult
)
from mcp.server.stdio import stdio_server
from mcp.server.streamable_http import streamable_http_server

from agents.simple_text_agent import SimpleTextAgent


class SimpleTextAnalysisRequest(BaseModel):
    """Request model for text analysis."""
    text: str = Field(..., description="Text to analyze")
    analysis_type: str = Field(default="sentiment", description="Type of analysis to perform")


class SimpleTextAnalysisResponse(BaseModel):
    """Response model for text analysis."""
    result: str = Field(..., description="Analysis result")
    confidence: float = Field(..., description="Confidence score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SimpleTextAgentMCPServerV2:
    """MCP Server v2 for SimpleTextAgent using mcp.server.Server."""
    
    def __init__(self, model: str = "ollama:llama3.2:latest"):
        """Initialize the MCP server v2."""
        self.model = model
        self.simple_text_agent = SimpleTextAgent(model=model)
        self.server = Server("SimpleTextAgent Server v2")
        self._setup_server()
        logger.info(f"SimpleTextAgent MCP Server v2 initialized with model: {model}")
    
    def _setup_server(self):
        """Set up the MCP server with tools and capabilities."""
        # Set server capabilities
        self.server.set_capabilities(
            ServerCapabilities(
                tools=ToolsCapability(
                    listChanged=True,
                    tools=[
                        Tool(
                            name="analyze_text_sentiment",
                            description="Analyze sentiment of text",
                            inputSchema={
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string", "description": "Text to analyze"},
                                    "analysis_type": {"type": "string", "description": "Type of analysis"}
                                },
                                "required": ["text"]
                            }
                        ),
                        Tool(
                            name="extract_text_features",
                            description="Extract features from text",
                            inputSchema={
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string", "description": "Text to analyze"}
                                },
                                "required": ["text"]
                            }
                        ),
                        Tool(
                            name="comprehensive_text_analysis",
                            description="Perform comprehensive text analysis",
                            inputSchema={
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string", "description": "Text to analyze"}
                                },
                                "required": ["text"]
                            }
                        ),
                        Tool(
                            name="batch_analyze_texts",
                            description="Analyze multiple texts in batch",
                            inputSchema={
                                "type": "object",
                                "properties": {
                                    "texts": {"type": "array", "items": {"type": "string"}}
                                },
                                "required": ["texts"]
                            }
                        ),
                        Tool(
                            name="get_simple_text_agent_capabilities",
                            description="Get agent capabilities",
                            inputSchema={"type": "object"}
                        )
                    ]
                )
            )
        )
        
        # Register tool handlers
        self.server.tool("analyze_text_sentiment")(self._analyze_text_sentiment)
        self.server.tool("extract_text_features")(self._extract_text_features)
        self.server.tool("comprehensive_text_analysis")(self._comprehensive_text_analysis)
        self.server.tool("batch_analyze_texts")(self._batch_analyze_texts)
        self.server.tool("get_simple_text_agent_capabilities")(self._get_capabilities)
    
    async def _analyze_text_sentiment(self, request: ToolCallRequest) -> ToolCallResult:
        """Handle text sentiment analysis."""
        try:
            text = request.arguments.get("text", "")
            analysis_type = request.arguments.get("analysis_type", "sentiment")
            
            result = await self.simple_text_agent.analyze_text_sentiment(text, analysis_type)
            
            return ToolCallResult(
                content=[{"type": "text", "text": str(result)}],
                isError=False
            )
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return ToolCallResult(
                content=[{"type": "text", "text": f"Error: {str(e)}"}],
                isError=True
            )
    
    async def _extract_text_features(self, request: ToolCallRequest) -> ToolCallResult:
        """Handle text feature extraction."""
        try:
            text = request.arguments.get("text", "")
            result = await self.simple_text_agent.extract_text_features(text)
            
            return ToolCallResult(
                content=[{"type": "text", "text": str(result)}],
                isError=False
            )
        except Exception as e:
            logger.error(f"Error in feature extraction: {e}")
            return ToolCallResult(
                content=[{"type": "text", "text": f"Error: {str(e)}"}],
                isError=True
            )
    
    async def _comprehensive_text_analysis(self, request: ToolCallRequest) -> ToolCallResult:
        """Handle comprehensive text analysis."""
        try:
            text = request.arguments.get("text", "")
            result = await self.simple_text_agent.comprehensive_text_analysis(text)
            
            return ToolCallResult(
                content=[{"type": "text", "text": str(result)}],
                isError=False
            )
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return ToolCallResult(
                content=[{"type": "text", "text": f"Error: {str(e)}"}],
                isError=True
            )
    
    async def _batch_analyze_texts(self, request: ToolCallRequest) -> ToolCallResult:
        """Handle batch text analysis."""
        try:
            texts = request.arguments.get("texts", [])
            result = await self.simple_text_agent.batch_analyze_texts(texts)
            
            return ToolCallResult(
                content=[{"type": "text", "text": str(result)}],
                isError=False
            )
        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            return ToolCallResult(
                content=[{"type": "text", "text": f"Error: {str(e)}"}],
                isError=True
            )
    
    async def _get_capabilities(self, request: ToolCallRequest) -> ToolCallResult:
        """Get agent capabilities."""
        try:
            capabilities = self.simple_text_agent.metadata.get("capabilities", [])
            result = f"Available capabilities: {', '.join(capabilities)}"
            
            return ToolCallResult(
                content=[{"type": "text", "text": result}],
                isError=False
            )
        except Exception as e:
            logger.error(f"Error getting capabilities: {e}")
            return ToolCallResult(
                content=[{"type": "text", "text": f"Error: {str(e)}"}],
                isError=True
            )
    
    def run_stdio(self):
        """Run the server using stdio transport."""
        return self.server.run(stdio_server())
    
    def run_http(self, host: str = "localhost", port: int = 8000):
        """Run the server using HTTP transport."""
        return self.server.run(streamable_http_server(host=host, port=port))


def create_simple_text_agent_mcp_server_v2(model: str = "ollama:llama3.2:latest") -> SimpleTextAgentMCPServerV2:
    """Create a SimpleTextAgent MCP Server v2 instance."""
    return SimpleTextAgentMCPServerV2(model=model)


if __name__ == "__main__":
    # Example usage
    server = create_simple_text_agent_mcp_server_v2()
    print("SimpleTextAgent MCP Server v2 created successfully!")
    print("Available tools:", [tool.name for tool in server.server.tools])
    
    # Run with stdio transport
    # server.run_stdio()
    
    # Or run with HTTP transport
    # server.run_http()
