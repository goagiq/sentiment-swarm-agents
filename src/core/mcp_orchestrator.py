"""
MCP Orchestrator implementation based on sentiment-swarm-agents pattern.
This follows the proper MCP tool integration and swarm orchestration approach.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional
from loguru import logger

# Import the mock Strands implementation
from src.core.strands_mock import Agent, Swarm, tool


class MCPOrchestrator:
    """MCP Orchestrator following the sentiment-swarm-agents pattern."""
    
    def __init__(self):
        self.agents = {}
        self.swarms = {}
        self.tools = {}
        self.mcp_servers = {}
        
        logger.info("Initializing MCP Orchestrator")
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default MCP tools following the pattern."""
        
        @tool(name="analyze_sentiment", description="Analyze sentiment of text content")
        def analyze_sentiment(text: str) -> Dict[str, Any]:
            """Analyze sentiment of the given text."""
            try:
                # Enhanced sentiment analysis
                positive_words = ["love", "great", "amazing", "wonderful", "excellent", "fantastic", "perfect"]
                negative_words = ["hate", "terrible", "awful", "bad", "horrible", "disgusting", "worst"]
                
                text_lower = text.lower()
                positive_count = sum(1 for word in positive_words if word in text_lower)
                negative_count = sum(1 for word in negative_words if word in text_lower)
                
                if positive_count > negative_count:
                    sentiment = "positive"
                    confidence = min(0.9, 0.7 + (positive_count * 0.1))
                elif negative_count > positive_count:
                    sentiment = "negative"
                    confidence = min(0.9, 0.7 + (negative_count * 0.1))
                else:
                    sentiment = "neutral"
                    confidence = 0.7
                
                return {
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "text": text,
                    "positive_score": positive_count,
                    "negative_score": negative_count
                }
            except Exception as e:
                logger.error(f"Error in sentiment analysis: {e}")
                return {"error": str(e)}
        
        @tool(name="extract_entities", description="Extract entities from text content")
        def extract_entities(text: str) -> Dict[str, Any]:
            """Extract entities from the given text."""
            try:
                entities = []
                import re
                
                # Enhanced entity patterns
                patterns = {
                    "PERSON": [
                        r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
                        r'\b[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+\b'
                    ],
                    "ORGANIZATION": [
                        r'\b[A-Z][A-Z]+\b',
                        r'\b[A-Z][a-z]+ (Inc|Corp|LLC|Ltd|Company|Organization)\b'
                    ],
                    "LOCATION": [
                        r'\b[A-Z][a-z]+, [A-Z]{2}\b',
                        r'\b[A-Z][a-z]+ (Street|Avenue|Road|Boulevard)\b'
                    ],
                    "EMAIL": [
                        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                    ],
                    "PHONE": [
                        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
                    ]
                }
                
                for entity_type, pattern_list in patterns.items():
                    for pattern in pattern_list:
                        matches = re.findall(pattern, text)
                        for match in matches:
                            entities.append({
                                "text": match,
                                "type": entity_type,
                                "confidence": 0.8
                            })
                
                return {
                    "entities": entities,
                    "text": text,
                    "total_entities": len(entities)
                }
            except Exception as e:
                logger.error(f"Error in entity extraction: {e}")
                return {"error": str(e)}
        
        @tool(name="process_image", description="Process and analyze image content")
        def process_image(image_path: str) -> Dict[str, Any]:
            """Process and analyze the given image."""
            try:
                # Mock image processing with enhanced features
                return {
                    "image_path": image_path,
                    "objects_detected": ["person", "car", "building", "tree"],
                    "confidence": 0.85,
                    "processing_time": 1.2,
                    "image_size": "1920x1080",
                    "format": "JPEG",
                    "analysis_type": "object_detection"
                }
            except Exception as e:
                logger.error(f"Error in image processing: {e}")
                return {"error": str(e)}
        
        @tool(name="analyze_audio", description="Analyze audio content and extract features")
        def analyze_audio(audio_path: str) -> Dict[str, Any]:
            """Analyze audio content."""
            try:
                return {
                    "audio_path": audio_path,
                    "duration": 120.5,
                    "format": "MP3",
                    "sample_rate": 44100,
                    "channels": 2,
                    "features": {
                        "tempo": 120,
                        "energy": 0.7,
                        "valence": 0.6,
                        "arousal": 0.5
                    }
                }
            except Exception as e:
                logger.error(f"Error in audio analysis: {e}")
                return {"error": str(e)}
        
        # Register tools
        self.tools = {
            "analyze_sentiment": analyze_sentiment,
            "extract_entities": extract_entities,
            "process_image": process_image,
            "analyze_audio": analyze_audio
        }
        
        logger.info(f"Registered {len(self.tools)} MCP tools")
    
    def create_sentiment_agent(self) -> Agent:
        """Create a sentiment analysis agent with MCP tools."""
        agent = Agent(
            name="sentiment_analyzer",
            system_prompt="You are a sentiment analysis expert. Use the available tools to analyze text sentiment and extract entities.",
            tools=[self.tools["analyze_sentiment"], self.tools["extract_entities"]]
        )
        
        self.agents["sentiment_analyzer"] = agent
        logger.info("Created sentiment analysis agent")
        return agent
    
    def create_vision_agent(self) -> Agent:
        """Create a vision analysis agent with MCP tools."""
        agent = Agent(
            name="vision_analyzer",
            system_prompt="You are a computer vision expert. Use the available tools to analyze images and extract visual features.",
            tools=[self.tools["process_image"]]
        )
        
        self.agents["vision_analyzer"] = agent
        logger.info("Created vision analysis agent")
        return agent
    
    def create_audio_agent(self) -> Agent:
        """Create an audio analysis agent with MCP tools."""
        agent = Agent(
            name="audio_analyzer",
            system_prompt="You are an audio analysis expert. Use the available tools to analyze audio content and extract features.",
            tools=[self.tools["analyze_audio"]]
        )
        
        self.agents["audio_analyzer"] = agent
        logger.info("Created audio analysis agent")
        return agent
    
    def create_report_agent(self) -> Agent:
        """Create a report generation agent."""
        agent = Agent(
            name="report_writer",
            system_prompt="You are a report writing expert. Create comprehensive reports based on analysis results.",
            tools=[self.tools["analyze_sentiment"], self.tools["extract_entities"]]
        )
        
        self.agents["report_writer"] = agent
        logger.info("Created report writing agent")
        return agent
    
    def create_analysis_swarm(self) -> Swarm:
        """Create a swarm for comprehensive analysis."""
        # Create specialized agents
        sentiment_agent = self.create_sentiment_agent()
        vision_agent = self.create_vision_agent()
        audio_agent = self.create_audio_agent()
        report_agent = self.create_report_agent()
        
        # Create swarm
        swarm = Swarm([sentiment_agent, vision_agent, audio_agent, report_agent])
        self.swarms["analysis_swarm"] = swarm
        
        logger.info("Created analysis swarm with 4 agents")
        return swarm
    
    async def run_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Run sentiment analysis using the sentiment agent."""
        try:
            agent = self.agents.get("sentiment_analyzer")
            if not agent:
                agent = self.create_sentiment_agent()
            
            prompt = f"Analyze the sentiment and extract entities from this text: {text}"
            result = await agent.run(prompt)
            
            return {
                "text": text,
                "analysis_result": result,
                "agent_used": "sentiment_analyzer"
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {"error": str(e)}
    
    async def run_image_analysis(self, image_path: str) -> Dict[str, Any]:
        """Run image analysis using the vision agent."""
        try:
            agent = self.agents.get("vision_analyzer")
            if not agent:
                agent = self.create_vision_agent()
            
            prompt = f"Analyze this image: {image_path}"
            result = await agent.run(prompt)
            
            return {
                "image_path": image_path,
                "analysis_result": result,
                "agent_used": "vision_analyzer"
            }
        except Exception as e:
            logger.error(f"Error in image analysis: {e}")
            return {"error": str(e)}
    
    async def run_swarm_analysis(self, content: str) -> Dict[str, Any]:
        """Run comprehensive analysis using the swarm."""
        try:
            swarm = self.swarms.get("analysis_swarm")
            if not swarm:
                swarm = self.create_analysis_swarm()
            
            prompt = f"Perform comprehensive analysis of this content: {content}"
            result = await swarm.run(prompt)
            
            return {
                "content": content,
                "swarm_result": result,
                "agents_used": len(swarm.agents)
            }
        except Exception as e:
            logger.error(f"Error in swarm analysis: {e}")
            return {"error": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "agents": {name: {"name": agent.name, "tools_count": len(agent.tools)} 
                      for name, agent in self.agents.items()},
            "swarms": {name: {"name": name, "agents_count": len(swarm.agents)} 
                      for name, swarm in self.swarms.items()},
            "tools": list(self.tools.keys()),
            "total_agents": len(self.agents),
            "total_swarms": len(self.swarms)
        }


# Global orchestrator instance
mcp_orchestrator = MCPOrchestrator()


# Helper functions for easy access
def get_sentiment_agent() -> Agent:
    """Get or create sentiment analysis agent."""
    return mcp_orchestrator.agents.get("sentiment_analyzer") or mcp_orchestrator.create_sentiment_agent()


def get_vision_agent() -> Agent:
    """Get or create vision analysis agent."""
    return mcp_orchestrator.agents.get("vision_analyzer") or mcp_orchestrator.create_vision_agent()


def get_analysis_swarm() -> Swarm:
    """Get or create analysis swarm."""
    return mcp_orchestrator.swarms.get("analysis_swarm") or mcp_orchestrator.create_analysis_swarm()


async def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze sentiment using the orchestrator."""
    return await mcp_orchestrator.run_sentiment_analysis(text)


async def analyze_image(image_path: str) -> Dict[str, Any]:
    """Analyze image using the orchestrator."""
    return await mcp_orchestrator.run_image_analysis(image_path)


async def run_comprehensive_analysis(content: str) -> Dict[str, Any]:
    """Run comprehensive analysis using the swarm."""
    return await mcp_orchestrator.run_swarm_analysis(content)
