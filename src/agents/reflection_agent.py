#!/usr/bin/env python3
"""
Reflection Coordinator Agent for Phase 5: Centralized agent reflection and communication.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import asyncio

from src.agents.base_agent import BaseAgent


class AgentCommunicationManager:
    """Manages inter-agent communication and questioning."""
    
    def __init__(self):
        self.agent_connections = {}
        self.communication_history = []
        self.question_queue = []
        self.logger = logging.getLogger(__name__)
    
    async def register_agent(self, agent_id: str, capabilities: Dict[str, Any]):
        """Register an agent for communication."""
        self.agent_connections[agent_id] = {
            "capabilities": capabilities,
            "status": "available",
            "last_communication": datetime.now().isoformat()
        }
        self.logger.info(f"Registered agent {agent_id} for communication")
    
    async def send_question(self, source_agent: str, target_agent: str, 
                          question: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a question from one agent to another."""
        try:
            if target_agent not in self.agent_connections:
                return {
                    "status": "error",
                    "error": f"Target agent {target_agent} not found"
                }
            
            question_id = f"q_{len(self.communication_history)}_{source_agent}_{target_agent}"
            
            question_data = {
                "question_id": question_id,
                "source_agent": source_agent,
                "target_agent": target_agent,
                "question": question,
                "context": context or {},
                "timestamp": datetime.now().isoformat(),
                "status": "pending"
            }
            
            self.question_queue.append(question_data)
            self.communication_history.append(question_data)
            
            return {
                "status": "success",
                "question_id": question_id,
                "message": f"Question sent to {target_agent}"
            }
            
        except Exception as e:
            self.logger.error(f"Error sending question: {e}")
            return {"status": "error", "error": str(e)}
    
    async def get_agent_feedback(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get feedback from a specific agent."""
        try:
            feedback = []
            for comm in self.communication_history:
                if comm.get("target_agent") == agent_id:
                    feedback.append(comm)
            return feedback
            
        except Exception as e:
            self.logger.error(f"Error getting agent feedback: {e}")
            return []


class ResponseValidator:
    """Validates and improves response quality."""
    
    def __init__(self):
        self.validation_criteria = {
            "accuracy": self._validate_accuracy,
            "completeness": self._validate_completeness,
            "relevance": self._validate_relevance,
            "consistency": self._validate_consistency
        }
        self.logger = logging.getLogger(__name__)
    
    async def validate_response(self, response: Dict[str, Any], 
                              criteria: List[str] = None) -> Dict[str, Any]:
        """Validate response against specified criteria."""
        try:
            if criteria is None:
                criteria = ["accuracy", "completeness", "relevance"]
            
            validation_results = {}
            overall_score = 0.0
            
            for criterion in criteria:
                if criterion in self.validation_criteria:
                    score = await self.validation_criteria[criterion](response)
                    validation_results[criterion] = score
                    overall_score += score
            
            overall_score /= len(criteria)
            
            return {
                "status": "success",
                "validation_results": validation_results,
                "overall_score": overall_score,
                "criteria_checked": criteria,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error validating response: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _validate_accuracy(self, response: Dict[str, Any]) -> float:
        """Validate response accuracy."""
        try:
            # Simple accuracy validation based on response structure
            score = 0.5  # Base score
            
            # Check for required fields
            if "status" in response:
                score += 0.2
            if "result" in response or "data" in response:
                score += 0.2
            if "timestamp" in response:
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error in accuracy validation: {e}")
            return 0.0
    
    async def _validate_completeness(self, response: Dict[str, Any]) -> float:
        """Validate response completeness."""
        try:
            score = 0.5  # Base score
            
            # Check response length and content
            response_str = str(response)
            if len(response_str) > 100:
                score += 0.3
            if len(response_str) > 500:
                score += 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error in completeness validation: {e}")
            return 0.0
    
    async def _validate_relevance(self, response: Dict[str, Any]) -> float:
        """Validate response relevance."""
        try:
            score = 0.5  # Base score
            
            # Check if response contains relevant keywords
            relevant_keywords = ["success", "result", "data", "analysis", "insight"]
            response_str = str(response).lower()
            
            keyword_count = sum(1 for keyword in relevant_keywords if keyword in response_str)
            score += min(keyword_count * 0.1, 0.5)
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error in relevance validation: {e}")
            return 0.0
    
    async def _validate_consistency(self, response: Dict[str, Any]) -> float:
        """Validate response consistency."""
        try:
            score = 0.5  # Base score
            
            # Check for internal consistency
            if "status" in response and "error" in response:
                if response["status"] == "success" and response["error"]:
                    score -= 0.3  # Inconsistent
                elif response["status"] == "error" and not response["error"]:
                    score -= 0.3  # Inconsistent
            
            return max(score, 0.0)
            
        except Exception as e:
            self.logger.error(f"Error in consistency validation: {e}")
            return 0.0


class ReflectionCoordinatorAgent(BaseAgent):
    """Central coordinator for agent reflection and communication."""
    
    def __init__(self):
        super().__init__()
        self.agent_id = "reflection_coordinator_agent"
        self.name = "Reflection Coordinator Agent"
        self.description = "Centralized agent reflection and communication"
        
        # Initialize components
        self.communication_manager = AgentCommunicationManager()
        self.response_validator = ResponseValidator()
        
        # Setup directories
        self.reflection_dir = Path("Results/reflection")
        self.reflection_dir.mkdir(parents=True, exist_ok=True)
        
        # Reflection state
        self.active_reflections = {}
        self.reflection_history = []
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Reflection Coordinator Agent initialized with reflection directory: {self.reflection_dir}")
    
    def can_process(self, content: str) -> bool:
        """Check if this agent can process the given content."""
        return True  # Can coordinate reflection for any content type
    
    async def process(self, content: str, **kwargs) -> Dict[str, Any]:
        """Process content and coordinate reflection."""
        try:
            query = kwargs.get("query", content)
            initial_response = kwargs.get("initial_response", {})
            reflection_type = kwargs.get("reflection_type", "comprehensive")
            
            # Coordinate reflection
            reflection_result = await self.coordinate_agent_reflection(
                query, initial_response, reflection_type
            )
            
            return {
                "status": "success",
                "agent_id": self.agent_id,
                "reflection_result": reflection_result,
                "query": query,
                "reflection_type": reflection_type
            }
            
        except Exception as e:
            return {
                "status": "error",
                "agent_id": self.agent_id,
                "error": str(e)
            }
    
    async def coordinate_agent_reflection(self, query: str, initial_response: Dict[str, Any],
                                        reflection_type: str = "comprehensive",
                                        include_agent_questioning: bool = True) -> Dict[str, Any]:
        """Coordinate agent reflection and communication."""
        try:
            reflection_id = f"reflection_{len(self.reflection_history)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Start reflection session
            reflection_session = {
                "reflection_id": reflection_id,
                "query": query,
                "initial_response": initial_response,
                "reflection_type": reflection_type,
                "start_time": datetime.now().isoformat(),
                "status": "active",
                "agent_feedback": [],
                "validation_results": {},
                "final_response": None
            }
            
            self.active_reflections[reflection_id] = reflection_session
            
            # Validate initial response
            validation_results = await self.response_validator.validate_response(
                initial_response, ["accuracy", "completeness", "relevance"]
            )
            reflection_session["validation_results"] = validation_results
            
            # Agent questioning if enabled
            if include_agent_questioning:
                agent_questions = await self._generate_agent_questions(
                    query, initial_response, reflection_type
                )
                reflection_session["agent_feedback"] = agent_questions
            
            # Generate final response
            final_response = await self._generate_final_response(
                initial_response, reflection_session
            )
            reflection_session["final_response"] = final_response
            reflection_session["status"] = "completed"
            reflection_session["end_time"] = datetime.now().isoformat()
            
            # Store in history
            self.reflection_history.append(reflection_session)
            del self.active_reflections[reflection_id]
            
            return {
                "status": "success",
                "reflection_id": reflection_id,
                "initial_response": initial_response,
                "validation_results": validation_results,
                "agent_feedback": reflection_session["agent_feedback"],
                "final_response": final_response,
                "reflection_type": reflection_type,
                "duration": "completed"
            }
            
        except Exception as e:
            self.logger.error(f"Error coordinating reflection: {e}")
            return {"status": "error", "error": str(e)}
    
    async def agent_questioning_system(self, source_agent: str, target_agent: str,
                                     question: str, context: Dict[str, Any] = None,
                                     response_format: str = "structured") -> Dict[str, Any]:
        """Enable agents to question and validate each other."""
        try:
            # Send question through communication manager
            question_result = await self.communication_manager.send_question(
                source_agent, target_agent, question, context
            )
            
            if question_result["status"] == "success":
                # Simulate response from target agent
                response = await self._simulate_agent_response(target_agent, question, context)
                
                return {
                    "status": "success",
                    "question_id": question_result["question_id"],
                    "source_agent": source_agent,
                    "target_agent": target_agent,
                    "question": question,
                    "response": response,
                    "response_format": response_format,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return question_result
                
        except Exception as e:
            self.logger.error(f"Error in agent questioning: {e}")
            return {"status": "error", "error": str(e)}
    
    async def get_reflection_insights(self, query_id: str,
                                    include_agent_feedback: bool = True,
                                    include_confidence_improvements: bool = True) -> Dict[str, Any]:
        """Get reflection insights and recommendations."""
        try:
            # Find reflection by query_id
            reflection = None
            for ref in self.reflection_history:
                if ref.get("reflection_id") == query_id:
                    reflection = ref
                    break
            
            if not reflection:
                return {"status": "error", "error": f"Reflection {query_id} not found"}
            
            insights = {
                "reflection_id": query_id,
                "query": reflection["query"],
                "reflection_type": reflection["reflection_type"],
                "validation_score": reflection["validation_results"].get("overall_score", 0.0),
                "duration": "completed"
            }
            
            if include_agent_feedback:
                insights["agent_feedback"] = reflection["agent_feedback"]
            
            if include_confidence_improvements:
                initial_score = reflection["validation_results"].get("overall_score", 0.0)
                final_score = 0.8  # Simulated improvement
                insights["confidence_improvement"] = {
                    "initial_score": initial_score,
                    "final_score": final_score,
                    "improvement": final_score - initial_score
                }
            
            return {
                "status": "success",
                "insights": insights,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting reflection insights: {e}")
            return {"status": "error", "error": str(e)}
    
    async def validate_response_quality(self, response: Dict[str, Any],
                                      validation_criteria: List[str] = None,
                                      include_improvement_suggestions: bool = True) -> Dict[str, Any]:
        """Validate and improve response quality."""
        try:
            if validation_criteria is None:
                validation_criteria = ["accuracy", "completeness", "relevance"]
            
            # Validate response
            validation_results = await self.response_validator.validate_response(
                response, validation_criteria
            )
            
            result = {
                "status": "success",
                "validation_results": validation_results,
                "response": response,
                "criteria_checked": validation_criteria
            }
            
            # Add improvement suggestions if requested
            if include_improvement_suggestions:
                suggestions = await self._generate_improvement_suggestions(
                    response, validation_results
                )
                result["improvement_suggestions"] = suggestions
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error validating response quality: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _generate_agent_questions(self, query: str, initial_response: Dict[str, Any],
                                      reflection_type: str) -> List[Dict[str, Any]]:
        """Generate questions for other agents to validate the response."""
        try:
            questions = []
            
            # Generate questions based on reflection type
            if reflection_type == "comprehensive":
                questions = [
                    {
                        "source_agent": self.agent_id,
                        "target_agent": "knowledge_graph_agent",
                        "question": f"Is the response consistent with our knowledge base for: {query}?",
                        "context": {"initial_response": initial_response}
                    },
                    {
                        "source_agent": self.agent_id,
                        "target_agent": "unified_text_agent",
                        "question": f"Can you validate the accuracy of this response: {str(initial_response)[:100]}...",
                        "context": {"query": query}
                    }
                ]
            elif reflection_type == "quick":
                questions = [
                    {
                        "source_agent": self.agent_id,
                        "target_agent": "knowledge_graph_agent",
                        "question": f"Quick validation needed for: {query}",
                        "context": {"initial_response": initial_response}
                    }
                ]
            
            return questions
            
        except Exception as e:
            self.logger.error(f"Error generating agent questions: {e}")
            return []
    
    async def _generate_final_response(self, initial_response: Dict[str, Any],
                                     reflection_session: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final response based on reflection and validation."""
        try:
            # Start with initial response
            final_response = initial_response.copy()
            
            # Apply improvements based on validation results
            validation_score = reflection_session["validation_results"].get("overall_score", 0.0)
            
            if validation_score < 0.7:
                # Add confidence warning
                final_response["confidence_warning"] = "Response quality below optimal threshold"
                final_response["suggested_improvements"] = [
                    "Consider additional validation",
                    "Cross-reference with knowledge base",
                    "Request human review for critical decisions"
                ]
            
            # Add reflection metadata
            final_response["reflection_metadata"] = {
                "reflection_id": reflection_session["reflection_id"],
                "validation_score": validation_score,
                "agent_feedback_count": len(reflection_session["agent_feedback"]),
                "reflection_type": reflection_session["reflection_type"]
            }
            
            return final_response
            
        except Exception as e:
            self.logger.error(f"Error generating final response: {e}")
            return initial_response
    
    async def _simulate_agent_response(self, target_agent: str, question: str,
                                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Simulate response from target agent."""
        try:
            # Simulate agent response based on agent type
            if "knowledge_graph" in target_agent:
                return {
                    "agent_id": target_agent,
                    "response": f"Knowledge graph validation for: {question}",
                    "confidence": 0.85,
                    "validation_result": "consistent"
                }
            elif "text" in target_agent:
                return {
                    "agent_id": target_agent,
                    "response": f"Text analysis validation for: {question}",
                    "confidence": 0.78,
                    "validation_result": "accurate"
                }
            else:
                return {
                    "agent_id": target_agent,
                    "response": f"General validation for: {question}",
                    "confidence": 0.7,
                    "validation_result": "acceptable"
                }
                
        except Exception as e:
            self.logger.error(f"Error simulating agent response: {e}")
            return {"error": str(e)}
    
    async def _generate_improvement_suggestions(self, response: Dict[str, Any],
                                              validation_results: Dict[str, Any]) -> List[str]:
        """Generate improvement suggestions based on validation results."""
        try:
            suggestions = []
            overall_score = validation_results.get("overall_score", 0.0)
            
            if overall_score < 0.6:
                suggestions.append("Consider adding more detailed analysis")
                suggestions.append("Include supporting evidence or references")
                suggestions.append("Request additional context or clarification")
            elif overall_score < 0.8:
                suggestions.append("Add confidence scores to key findings")
                suggestions.append("Include alternative interpretations")
            else:
                suggestions.append("Response quality is good, consider adding metadata")
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating improvement suggestions: {e}")
            return ["Unable to generate suggestions due to error"]
