"""
Text agent swarm for coordinated sentiment analysis using Strands framework.
"""

import asyncio
from typing import List, Optional

from loguru import logger
from src.core.strands_mock import tool, Agent, Swarm

from agents.base_agent import BaseAgent
from agents.text_agent import TextAgent
from config.config import config
from core.models import (
    AnalysisRequest, AnalysisResult, DataType, SentimentResult
)


class TextAgentSwarm(BaseAgent):
    """Swarm of text agents for coordinated sentiment analysis using Strands."""
    
    def __init__(
        self, 
        agent_count: int = 3, 
        model_name: Optional[str] = None,
        **kwargs
    ):
        # Use config system instead of hardcoded values
        default_model = config.model.default_text_model
        super().__init__(
            model_name=model_name or default_model, 
            **kwargs
        )
        self.agent_count = agent_count
        self.text_agents: List[TextAgent] = []
        self.metadata["swarm_size"] = agent_count
        self.metadata["model"] = model_name or default_model
        self.metadata["capabilities"] = [
            "text", "swarm", "coordinated_analysis"
        ]
        
        # Initialize text agents
        self._initialize_agents()
        
        # Initialize Strands Swarm for coordination
        self._initialize_strands_swarm()
    
    def _get_tools(self) -> list:
        """Get list of tools for this swarm."""
        return [
            self.coordinate_sentiment_analysis,
            self.analyze_text_with_swarm,
            self.get_swarm_status,
            self.distribute_workload
        ]
    
    def _initialize_agents(self):
        """Initialize the text agents in the swarm."""
        for i in range(self.agent_count):
            agent = TextAgent(
                agent_id=f"text_agent_{i+1}",
                model_name=self.metadata["model"]
            )
            self.text_agents.append(agent)
        
        logger.info(f"Initialized {self.agent_count} text agents in swarm")
    
    def _initialize_strands_swarm(self):
        """Initialize the Strands Swarm for coordination."""
        # Create specialized Strands agents for the swarm
        swarm_agents = []
        
        # Create a researcher agent for initial analysis
        researcher = Agent(
            name="researcher",
            system_prompt="""You are a text sentiment research specialist. 
            Analyze the given text and identify key sentiment indicators, 
            emotional cues, and context that could affect sentiment analysis.
            Focus on understanding the content before passing to sentiment experts."""
        )
        
        # Create a sentiment specialist agent
        sentiment_specialist = Agent(
            name="sentiment_specialist", 
            system_prompt="""You are a sentiment analysis expert. 
            Use the available tools to perform detailed sentiment analysis.
            Always coordinate with other agents in the swarm for comprehensive results."""
        )
        
        # Create a reviewer agent for quality assurance
        reviewer = Agent(
            name="reviewer",
            system_prompt="""You are a sentiment analysis reviewer. 
            Review and validate sentiment analysis results from other agents.
            Ensure consistency and accuracy across the swarm's analysis."""
        )
        
        # Create a coordinator agent that manages the workflow
        coordinator = Agent(
            name="coordinator",
            system_prompt="""You are a sentiment analysis swarm coordinator. 
            Coordinate the workflow between researcher, sentiment specialist, and reviewer.
            Use the available tools to manage the analysis process and aggregate results.
            Always ensure proper handoffs between specialized agents.""",
            tools=self._get_tools()
        )
        
        swarm_agents = [researcher, sentiment_specialist, reviewer, coordinator]
        
        # Create a properly configured Strands Swarm according to documentation
        self.strands_swarm = Swarm(
            swarm_agents,
            max_handoffs=20,
            max_iterations=20,
            execution_timeout=900.0,  # 15 minutes
            node_timeout=300.0,       # 5 minutes per agent
            repetitive_handoff_detection_window=8,  # Check last 8 handoffs
            repetitive_handoff_min_unique_agents=3  # Require at least 3 unique agents
        )
        
        logger.info(f"Initialized Strands Swarm with {len(swarm_agents)} specialized agents")
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this swarm can process the request."""
        return request.data_type in [DataType.TEXT, DataType.SOCIAL_MEDIA]
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process text analysis request using Strands Swarm coordination."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Extract text content
            text_content = self._extract_text(request.content)
            
            # Use Strands Swarm to coordinate the analysis
            prompt = (
                f"Coordinate sentiment analysis for this text: {text_content}\n\n"
                f"Workflow:\n"
                f"1. Researcher: Analyze the text content and identify key indicators\n"
                f"2. Sentiment Specialist: Perform detailed sentiment analysis using tools\n"
                f"3. Reviewer: Validate and ensure quality of results\n"
                f"4. Coordinator: Aggregate final results and provide comprehensive analysis\n\n"
                f"Use the available tools and coordinate between all agents in the swarm."
            )
            
            response = await self.strands_swarm.invoke_async(prompt)
            
            # Parse the response and create sentiment result
            sentiment_result = self._parse_swarm_response(str(response))
            
            # Create analysis result
            result = AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=sentiment_result,
                processing_time=0.0,  # Will be set by base class
                status=None,  # Will be set by base class
                raw_content=str(request.content),
                extracted_text=text_content,
                metadata={
                    "agent_id": self.agent_id,
                    "swarm_size": self.agent_count,
                    "method": "strands_swarm",
                    "model": self.metadata["model"],
                    "tools_used": [tool.__name__ for tool in self._get_tools()],
                    "strands_swarm_used": True,
                    "swarm_agents": ["researcher", "sentiment_specialist", "reviewer", "coordinator"],
                    "swarm_config": {
                        "max_handoffs": 20,
                        "max_iterations": 20,
                        "execution_timeout": 900.0,
                        "node_timeout": 300.0
                    }
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Swarm processing failed: {e}")
            # Return neutral sentiment on error
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="neutral",
                    confidence=0.0,
                    metadata={"error": str(e)}
                ),
                processing_time=asyncio.get_event_loop().time() - start_time,
                status="failed",
                raw_content=str(request.content),
                extracted_text="",
                metadata={
                    "agent_id": self.agent_id,
                    "swarm_size": self.agent_count,
                    "error": str(e)
                }
            )
    
    def _extract_text(self, content) -> str:
        """Extract text content from various input formats."""
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            # Handle social media posts, API responses, etc.
            if "text" in content:
                return content["text"]
            elif "content" in content:
                return content["content"]
            elif "message" in content:
                return content["message"]
            else:
                return str(content)
        else:
            return str(content)
    
    @tool
    async def coordinate_sentiment_analysis(self, text: str) -> dict:
        """Coordinate sentiment analysis across multiple text agents."""
        try:
            # Distribute the text to multiple agents for analysis
            tasks = []
            for agent in self.text_agents:
                task = agent.analyze_text_sentiment(text)
                tasks.append(task)
            
            # Wait for all agents to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Aggregate results
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Agent {i+1} failed: {result}")
                else:
                    valid_results.append(result)
            
            if not valid_results:
                raise RuntimeError("All agents failed to analyze text")
            
            # Use the most confident result
            best_result = max(
                valid_results, 
                key=lambda x: x.get("content", [{}])[0].get("json", {}).get("confidence", 0)
            )
            
            return {
                "status": "success",
                "content": [{
                    "json": {
                        "method": "swarm_coordination",
                        "agents_used": len(valid_results),
                        "total_agents": len(self.text_agents),
                        "best_result": best_result.get("content", [{}])[0].get("json", {}),
                        "coordination_success": True
                    }
                }]
            }
            
        except Exception as e:
            logger.error(f"Swarm coordination failed: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Swarm coordination error: {str(e)}"}]
            }
    
    @tool
    async def analyze_text_with_swarm(self, text: str) -> dict:
        """Analyze text using the entire swarm."""
        try:
            # Use the coordinate method
            return await self.coordinate_sentiment_analysis(text)
            
        except Exception as e:
            logger.error(f"Swarm analysis failed: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Swarm analysis error: {str(e)}"}]
            }
    
    @tool
    async def get_swarm_status(self) -> dict:
        """Get status of all agents in the swarm."""
        try:
            statuses = []
            for i, agent in enumerate(self.text_agents):
                agent_status = agent.get_status()
                statuses.append({
                    "agent_id": agent_status["agent_id"],
                    "status": agent_status["status"],
                    "current_load": agent_status["current_load"],
                    "max_capacity": agent_status["max_capacity"]
                })
            
            return {
                "status": "success",
                "content": [{
                    "json": {
                        "swarm_id": self.agent_id,
                        "total_agents": len(self.text_agents),
                        "agent_statuses": statuses,
                        "swarm_status": self.get_status(),
                        "strands_swarm_active": True
                    }
                }]
            }
            
        except Exception as e:
            logger.error(f"Failed to get swarm status: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Status error: {str(e)}"}]
            }
    
    @tool
    async def distribute_workload(self, texts: List[str]) -> dict:
        """Distribute multiple texts across the swarm."""
        try:
            if not texts:
                return {
                    "status": "error",
                    "content": [{"text": "No texts provided"}]
                }
            
            # Distribute texts across agents
            results = []
            for i, text in enumerate(texts):
                agent = self.text_agents[i % len(self.text_agents)]
                try:
                    result = await agent.analyze_text_sentiment(text)
                    results.append({
                        "text_index": i,
                        "agent_id": agent.agent_id,
                        "result": result
                    })
                except Exception as e:
                    results.append({
                        "text_index": i,
                        "agent_id": agent.agent_id,
                        "error": str(e)
                    })
            
            return {
                "status": "success",
                "content": [{
                    "json": {
                        "method": "workload_distribution",
                        "texts_processed": len(texts),
                        "agents_used": len(self.text_agents),
                        "results": results,
                        "distribution_success": True
                    }
                }]
            }
            
        except Exception as e:
            logger.error(f"Workload distribution failed: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Distribution error: {str(e)}"}]
            }
    
    def _parse_swarm_response(self, response: str) -> SentimentResult:
        """Parse the swarm response from Strands tools."""
        try:
            # Try to extract sentiment information from the response
            response_lower = response.lower()
            
            if "positive" in response_lower:
                label = "positive"
                confidence = 0.8
            elif "negative" in response_lower:
                label = "negative"
                confidence = 0.8
            else:
                label = "neutral"
                confidence = 0.6
            
            return SentimentResult(
                label=label,
                confidence=confidence,
                scores={
                    "positive": 0.8 if label == "positive" else 0.1,
                    "negative": 0.8 if label == "negative" else 0.1,
                    "neutral": 0.6 if label == "neutral" else 0.1
                },
                metadata={
                    "method": "strands_swarm",
                    "raw_response": response,
                    "swarm_size": self.agent_count,
                    "strands_swarm_used": True
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to parse swarm response: {e}")
            # Return neutral sentiment on parsing failure
            return SentimentResult(
                label="neutral",
                confidence=0.0,
                scores={"neutral": 1.0},
                metadata={
                    "method": "strands_swarm",
                    "error": str(e),
                    "raw_response": response,
                    "strands_swarm_used": True
                }
            )
    
    async def start(self):
        """Start the swarm and all agents."""
        await super().start()
        
        # Start all text agents
        for agent in self.text_agents:
            await agent.start()
        
        logger.info(f"Text agent swarm {self.agent_id} started")
    
    async def stop(self):
        """Stop the swarm and all agents."""
        # Stop all text agents
        for agent in self.text_agents:
            await agent.stop()
        
        await super().stop()
        logger.info(f"Text agent swarm {self.agent_id} stopped")
    
    def get_status(self) -> dict:
        """Get comprehensive swarm status."""
        base_status = super().get_status()
        
        # Add swarm-specific information
        agent_statuses = []
        for agent in self.text_agents:
            agent_statuses.append(agent.get_status())
        
        base_status.update({
            "swarm_size": self.agent_count,
            "agent_statuses": agent_statuses,
            "active_agents": sum(1 for a in self.text_agents if a.status == "running"),
            "strands_swarm_active": hasattr(self, 'strands_swarm')
        })
        
        return base_status
