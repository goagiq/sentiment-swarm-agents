"""
Base agent class for the agentic swarm using Strands framework.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from src.core.models import (
    AnalysisRequest,
    AnalysisResult,
    ProcessingStatus,
    SentimentResult,
)
try:
    from strands import Agent
except ImportError:
    from src.core.strands_mock import Agent

logger = logging.getLogger(__name__)


class StrandsBaseAgent(ABC):
    """Base class for all Strands-compliant agents in the swarm."""
    
    def __init__(
        self, 
        agent_id: Optional[str] = None, 
        max_capacity: int = 10, 
        model_name: str = "llama3.2:latest"
    ):
        self.agent_id = (
            agent_id or f"{self.__class__.__name__}_{uuid4().hex[:8]}"
        )
        self.max_capacity = max_capacity
        self.current_load = 0
        self.status = "idle"
        self.last_heartbeat = datetime.now(timezone.utc)
        self.metadata: Dict[str, Any] = {}
        self._shutdown_event = asyncio.Event()
        
        # Create Strands Agent with tools
        self.strands_agent = Agent(
            name=self.agent_id,
            model=model_name,
            tools=self._get_tools()
        )
        
        logger.info(f"Initialized Strands agent {self.agent_id}")
    
    def _get_tools(self) -> list:
        """Get list of tools for this agent. Override in subclasses."""
        return []
    
    @abstractmethod
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the given request."""
        pass
    
    @abstractmethod
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process the analysis request."""
        pass
    
    async def start(self):
        """Start the agent."""
        self.status = "running"
        logger.info(f"Agent {self.agent_id} started")
        
        # Start heartbeat loop
        asyncio.create_task(self._heartbeat_loop())
    
    async def stop(self):
        """Stop the agent."""
        self.status = "stopping"
        self._shutdown_event.set()
        logger.info(f"Agent {self.agent_id} stopping")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat updates."""
        while not self._shutdown_event.is_set():
            try:
                self.last_heartbeat = datetime.now(timezone.utc)
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
            except asyncio.CancelledError:
                break
        
        self.status = "stopped"
        logger.info(f"Agent {self.agent_id} stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.__class__.__name__,
            "status": self.status,
            "last_heartbeat": self.last_heartbeat,
            "current_load": self.current_load,
            "max_capacity": self.max_capacity,
            "metadata": self.metadata,
            "strands_agent": self.strands_agent.name,
            "tools_count": len(self._get_tools())
        }
    
    async def _acquire_slot(self) -> bool:
        """Try to acquire a processing slot."""
        if self.current_load >= self.max_capacity:
            return False
        
        self.current_load += 1
        return True
    
    def _release_slot(self):
        """Release a processing slot."""
        if self.current_load > 0:
            self.current_load -= 1
    
    async def _process_with_timing(
        self, request: AnalysisRequest
    ) -> AnalysisResult:
        """Process request with timing and error handling."""
        start_time = time.time()
        
        try:
            if not await self._acquire_slot():
                raise RuntimeError("Agent at maximum capacity")
            
            self.status = "processing"
            result = await self.process(request)
            
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            result.status = ProcessingStatus.COMPLETED
            
            logger.info(
                f"Agent {self.agent_id} processed request {request.id} "
                f"in {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                f"Agent {self.agent_id} failed to process request "
                f"{request.id}: {str(e)}"
            )
            
            # Create error result
            error_result = AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="neutral",
                    confidence=0.0,
                    metadata={"error": str(e)}
                ),
                processing_time=processing_time,
                status=ProcessingStatus.FAILED,
                metadata={"error": str(e), "agent_id": self.agent_id}
            )
            
            return error_result
            
        finally:
            self._release_slot()
            self.status = "idle" if self.current_load == 0 else "busy"
    
    async def process_request(
        self, request: AnalysisRequest
    ) -> AnalysisResult:
        """Process a request with proper timing and field population."""
        return await self._process_with_timing(request)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.agent_id})"
    
    def __repr__(self) -> str:
        return self.__str__()


# Keep the old BaseAgent for backward compatibility during transition
BaseAgent = StrandsBaseAgent
