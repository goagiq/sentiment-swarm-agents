#!/usr/bin/env python3
"""
Semantic Search Agent for Phase 5: Intelligent cross-modal semantic search.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import numpy as np

from src.agents.base_agent import BaseAgent


class SemanticSearchEngine:
    """Core semantic search engine for cross-modal content."""
    
    def __init__(self):
        self.search_index = {}
        self.embeddings_cache = {}
        self.agent_capabilities = {}
        self.logger = logging.getLogger(__name__)
        
    async def create_embeddings(self, content: str, content_type: str) -> List[float]:
        """Create embeddings for content using appropriate model."""
        try:
            # Placeholder for actual embedding generation
            # In production, use sentence-transformers or similar
            import hashlib
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            if content_hash in self.embeddings_cache:
                return self.embeddings_cache[content_hash]
            
            # Simulate embedding generation
            embedding = [float(x) for x in content_hash[:10]] + [0.0] * 382  # 392-dim vector
            self.embeddings_cache[content_hash] = embedding
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error creating embeddings: {e}")
            return [0.0] * 392
    
    async def semantic_similarity(self, query_embedding: List[float], content_embedding: List[float]) -> float:
        """Calculate semantic similarity between query and content."""
        try:
            query_vec = np.array(query_embedding)
            content_vec = np.array(content_embedding)
            
            # Cosine similarity
            similarity = np.dot(query_vec, content_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(content_vec))
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    async def search_content(self, query: str, content_types: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant content across specified types."""
        try:
            query_embedding = await self.create_embeddings(query, "text")
            results = []
            
            for content_type in content_types:
                if content_type in self.search_index:
                    for content_id, content_data in self.search_index[content_type].items():
                        similarity = await self.semantic_similarity(
                            query_embedding, 
                            content_data.get("embedding", [0.0] * 392)
                        )
                        
                        if similarity > 0.1:  # Threshold
                            results.append({
                                "content_id": content_id,
                                "content_type": content_type,
                                "similarity": similarity,
                                "content": content_data.get("content", ""),
                                "metadata": content_data.get("metadata", {})
                            })
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}")
            return []


class IntelligentRouter:
    """Intelligent routing system for optimal agent selection."""
    
    def __init__(self):
        self.agent_capabilities = {}
        self.performance_metrics = {}
        self.routing_history = []
        self.logger = logging.getLogger(__name__)
        
    async def register_agent_capabilities(self, agent_id: str, capabilities: Dict[str, Any]):
        """Register agent capabilities and specializations."""
        self.agent_capabilities[agent_id] = capabilities
        self.logger.info(f"Registered capabilities for agent {agent_id}")
    
    async def select_optimal_agents(self, query: str, content_data: Dict[str, Any], strategy: str = "accuracy") -> List[str]:
        """Select optimal agents based on query and content."""
        try:
            candidates = []
            
            for agent_id, capabilities in self.agent_capabilities.items():
                score = await self._calculate_agent_score(agent_id, query, content_data, strategy)
                candidates.append((agent_id, score))
            
            # Sort by score and return top agents
            candidates.sort(key=lambda x: x[1], reverse=True)
            return [agent_id for agent_id, score in candidates[:3]]  # Top 3 agents
            
        except Exception as e:
            self.logger.error(f"Error selecting optimal agents: {e}")
            return []
    
    async def _calculate_agent_score(self, agent_id: str, query: str, content_data: Dict[str, Any], strategy: str) -> float:
        """Calculate agent suitability score."""
        try:
            capabilities = self.agent_capabilities.get(agent_id, {})
            score = 0.0
            
            # Content type matching
            content_types = content_data.get("types", [])
            supported_types = capabilities.get("supported_types", [])
            
            type_match = len(set(content_types) & set(supported_types)) / max(len(content_types), 1)
            score += type_match * 0.4
            
            # Query complexity matching
            query_complexity = len(query.split()) / 10.0  # Simple heuristic
            agent_complexity = capabilities.get("complexity_handling", 0.5)
            complexity_match = 1.0 - abs(query_complexity - agent_complexity)
            score += complexity_match * 0.3
            
            # Performance metrics
            performance = self.performance_metrics.get(agent_id, {}).get("accuracy", 0.5)
            score += performance * 0.3
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating agent score: {e}")
            return 0.0


class ResultSynthesizer:
    """Synthesize results from multiple agents."""
    
    def __init__(self):
        self.synthesis_strategies = {
            "weighted": self._weighted_synthesis,
            "consensus": self._consensus_synthesis,
            "hierarchical": self._hierarchical_synthesis
        }
        self.logger = logging.getLogger(__name__)
    
    async def synthesize_results(self, results: List[Dict[str, Any]], strategy: str = "weighted") -> Dict[str, Any]:
        """Synthesize multiple agent results."""
        try:
            if strategy in self.synthesis_strategies:
                return await self.synthesis_strategies[strategy](results)
            else:
                return await self._weighted_synthesis(results)
                
        except Exception as e:
            self.logger.error(f"Error synthesizing results: {e}")
            return {"error": str(e)}
    
    async def _weighted_synthesis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Weighted synthesis based on agent confidence."""
        try:
            if not results:
                return {"synthesized_result": "No results available"}
            
            total_weight = 0
            weighted_sum = {}
            
            for result in results:
                weight = result.get("confidence", 0.5)
                total_weight += weight
                
                for key, value in result.items():
                    if key != "confidence":
                        if key not in weighted_sum:
                            weighted_sum[key] = 0
                        weighted_sum[key] += value * weight
            
            # Normalize by total weight
            if total_weight > 0:
                synthesized = {key: value / total_weight for key, value in weighted_sum.items()}
            else:
                synthesized = weighted_sum
            
            return {
                "synthesized_result": synthesized,
                "strategy": "weighted",
                "confidence": total_weight / len(results) if results else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error in weighted synthesis: {e}")
            return {"error": str(e)}
    
    async def _consensus_synthesis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consensus-based synthesis."""
        try:
            if not results:
                return {"synthesized_result": "No results available"}
            
            # Find common elements across results
            common_keys = set.intersection(*[set(result.keys()) for result in results])
            consensus_result = {}
            
            for key in common_keys:
                values = [result[key] for result in results]
                # Simple consensus: most common value
                consensus_result[key] = max(set(values), key=values.count)
            
            return {
                "synthesized_result": consensus_result,
                "strategy": "consensus",
                "confidence": len(common_keys) / max(len(results[0].keys()), 1) if results else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error in consensus synthesis: {e}")
            return {"error": str(e)}
    
    async def _hierarchical_synthesis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Hierarchical synthesis based on agent hierarchy."""
        try:
            if not results:
                return {"synthesized_result": "No results available"}
            
            # Sort by agent priority/confidence
            sorted_results = sorted(results, key=lambda x: x.get("confidence", 0.0), reverse=True)
            
            # Use highest confidence result as base, supplement with others
            base_result = sorted_results[0]
            synthesized = base_result.copy()
            
            # Supplement with additional information from other agents
            for result in sorted_results[1:]:
                for key, value in result.items():
                    if key not in synthesized:
                        synthesized[key] = value
            
            return {
                "synthesized_result": synthesized,
                "strategy": "hierarchical",
                "confidence": base_result.get("confidence", 0.0)
            }
            
        except Exception as e:
            self.logger.error(f"Error in hierarchical synthesis: {e}")
            return {"error": str(e)}


class SemanticSearchAgent(BaseAgent):
    """Agent for intelligent cross-modal semantic search and routing."""
    
    def __init__(self):
        super().__init__()
        self.agent_id = "semantic_search_agent"
        self.name = "Semantic Search Agent"
        self.description = "Intelligent cross-modal semantic search and routing"
        
        # Initialize components
        self.search_engine = SemanticSearchEngine()
        self.router = IntelligentRouter()
        self.synthesizer = ResultSynthesizer()
        
        # Setup directories
        self.search_dir = Path("Results/semantic_search")
        self.search_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Semantic Search Agent initialized with search directory: {self.search_dir}")
    
    def can_process(self, content: str) -> bool:
        """Check if this agent can process the given content."""
        return True  # Can process any content type for semantic search
    
    async def process(self, content: str, **kwargs) -> Dict[str, Any]:
        """Process content and perform semantic search."""
        try:
            query = kwargs.get("query", content)
            content_types = kwargs.get("content_types", ["text", "image", "audio", "video", "document"])
            search_strategy = kwargs.get("search_strategy", "accuracy")
            
            # Perform semantic search
            search_results = await self.search_engine.search_content(query, content_types)
            
            # Route to optimal agents
            optimal_agents = await self.router.select_optimal_agents(query, {"types": content_types}, search_strategy)
            
            # Synthesize results
            synthesized_result = await self.synthesizer.synthesize_results(search_results, "weighted")
            
            return {
                "status": "success",
                "agent_id": self.agent_id,
                "search_results": search_results,
                "optimal_agents": optimal_agents,
                "synthesized_result": synthesized_result,
                "query": query,
                "content_types": content_types,
                "search_strategy": search_strategy
            }
            
        except Exception as e:
            return {
                "status": "error",
                "agent_id": self.agent_id,
                "error": str(e)
            }
    
    async def semantic_search_intelligent(self, query: str, content_types: List[str] = None, 
                                        search_strategy: str = "accuracy", include_agent_metadata: bool = True,
                                        combine_results: bool = True) -> Dict[str, Any]:
        """Intelligent semantic search across all content types."""
        try:
            if content_types is None:
                content_types = ["text", "image", "audio", "video", "document"]
            
            # Perform search
            search_results = await self.search_engine.search_content(query, content_types)
            
            # Get agent metadata if requested
            agent_metadata = {}
            if include_agent_metadata:
                agent_metadata = self.router.agent_capabilities.copy()
            
            # Combine results if requested
            final_results = search_results
            if combine_results and len(search_results) > 1:
                synthesized = await self.synthesizer.synthesize_results(search_results, "weighted")
                final_results = [synthesized]
            
            return {
                "status": "success",
                "query": query,
                "content_types": content_types,
                "search_strategy": search_strategy,
                "results": final_results,
                "agent_metadata": agent_metadata,
                "total_results": len(search_results),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "query": query
            }
    
    async def route_query_intelligently(self, query: str, content_data: Dict[str, Any] = None,
                                      routing_strategy: str = "accuracy", include_fallback: bool = True) -> Dict[str, Any]:
        """Route queries to optimal agents based on content and capability."""
        try:
            if content_data is None:
                content_data = {}
            
            # Select optimal agents
            optimal_agents = await self.router.select_optimal_agents(query, content_data, routing_strategy)
            
            # Add fallback agents if requested
            fallback_agents = []
            if include_fallback and len(optimal_agents) < 2:
                fallback_agents = ["unified_text_agent", "knowledge_graph_agent"]
            
            return {
                "status": "success",
                "query": query,
                "routing_strategy": routing_strategy,
                "optimal_agents": optimal_agents,
                "fallback_agents": fallback_agents,
                "total_agents": len(optimal_agents) + len(fallback_agents),
                "content_data": content_data
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "query": query
            }
    
    async def combine_agent_results(self, results: List[Dict[str, Any]], 
                                  combination_strategy: str = "weighted",
                                  include_confidence_scores: bool = True) -> Dict[str, Any]:
        """Combine and synthesize results from multiple agents."""
        try:
            # Synthesize results
            synthesized = await self.synthesizer.synthesize_results(results, combination_strategy)
            
            # Add confidence scores if requested
            if include_confidence_scores:
                confidence_scores = [result.get("confidence", 0.5) for result in results]
                synthesized["confidence_analysis"] = {
                    "individual_scores": confidence_scores,
                    "average_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0,
                    "max_confidence": max(confidence_scores) if confidence_scores else 0.0,
                    "min_confidence": min(confidence_scores) if confidence_scores else 0.0
                }
            
            return {
                "status": "success",
                "combination_strategy": combination_strategy,
                "synthesized_result": synthesized,
                "input_results_count": len(results),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def get_agent_capabilities(self, agent_ids: List[str] = None,
                                   include_performance_metrics: bool = True) -> Dict[str, Any]:
        """Get agent capabilities and specializations."""
        try:
            if agent_ids is None:
                agent_ids = list(self.router.agent_capabilities.keys())
            
            capabilities = {}
            for agent_id in agent_ids:
                if agent_id in self.router.agent_capabilities:
                    capabilities[agent_id] = self.router.agent_capabilities[agent_id].copy()
                    
                    if include_performance_metrics and agent_id in self.router.performance_metrics:
                        capabilities[agent_id]["performance_metrics"] = self.router.performance_metrics[agent_id]
            
            return {
                "status": "success",
                "capabilities": capabilities,
                "total_agents": len(capabilities),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
