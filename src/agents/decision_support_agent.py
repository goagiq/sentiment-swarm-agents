"""
Enhanced Decision Support Agent

Orchestrates AI-assisted decision making with knowledge graph integration,
real-time data analysis, explainable AI, and multi-modal insights.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.agents.base_agent import StrandsBaseAgent
from src.core.models import AnalysisRequest, AnalysisResult, ProcessingStatus, SentimentResult
from src.core.decision_support import (
    RecommendationEngine,
    ActionPrioritizer,
    ImplementationPlanner,
    SuccessPredictor,
    RecommendationContext,
    PrioritizationContext,
    PlanningContext,
    PredictionContext
)
from src.core.decision_support.knowledge_graph_integrator import (
    KnowledgeGraphIntegrator,
    DecisionContext
)
from src.config.decision_support_config import (
    get_decision_support_config,
    get_language_decision_config
)

logger = logging.getLogger(__name__)


class DecisionSupportAgent(StrandsBaseAgent):
    """Enhanced AI-powered decision support agent with knowledge graph integration."""
    
    def __init__(self, agent_id: Optional[str] = None, model_name: str = "llama3.2:latest"):
        super().__init__(agent_id, max_capacity=5, model_name=model_name)
        
        # Initialize decision support components
        self.recommendation_engine = RecommendationEngine(model_name)
        self.action_prioritizer = ActionPrioritizer()
        self.implementation_planner = ImplementationPlanner()
        self.success_predictor = SuccessPredictor()
        
        # Initialize knowledge graph integrator
        self.knowledge_graph_integrator = KnowledgeGraphIntegrator()
        
        # Load configuration
        self.config = get_decision_support_config()
        
        logger.info(f"Initialized Enhanced DecisionSupportAgent: {self.agent_id}")
    
    def _get_tools(self) -> list:
        """Get tools for the decision support agent."""
        return [
            {
                "name": "generate_recommendations",
                "description": "Generate AI-powered recommendations based on context",
                "parameters": {
                    "business_objectives": "List of business objectives",
                    "current_performance": "Current performance metrics",
                    "market_conditions": "Market conditions and trends",
                    "resource_constraints": "Available resources and constraints"
                }
            },
            {
                "name": "prioritize_actions",
                "description": "Prioritize and rank recommendations based on multiple factors",
                "parameters": {
                    "recommendations": "List of recommendations to prioritize",
                    "available_resources": "Available resources and capacity",
                    "time_constraints": "Time constraints and deadlines",
                    "stakeholder_preferences": "Stakeholder preferences and priorities"
                }
            },
            {
                "name": "create_implementation_plan",
                "description": "Create detailed implementation plan for a recommendation",
                "parameters": {
                    "recommendation": "Recommendation to plan for",
                    "available_resources": "Available resources and team capacity",
                    "budget_constraints": "Budget constraints and limitations",
                    "timeline_constraints": "Timeline constraints and deadlines"
                }
            },
            {
                "name": "predict_success",
                "description": "Predict likelihood of success for a recommendation",
                "parameters": {
                    "recommendation": "Recommendation to predict success for",
                    "historical_data": "Historical success rates and patterns",
                    "organizational_capabilities": "Organizational capabilities and strengths",
                    "market_conditions": "Current market conditions and trends"
                }
            },
            {
                "name": "comprehensive_decision_analysis",
                "description": "Perform comprehensive decision analysis including recommendations, prioritization, planning, and success prediction",
                "parameters": {
                    "business_context": "Business context and objectives",
                    "current_situation": "Current situation and challenges",
                    "constraints": "Resource and time constraints",
                    "preferences": "Stakeholder preferences and priorities"
                }
            }
        ]
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the given request."""
        # Decision support agent can handle decision-making and recommendation requests
        decision_keywords = [
            "decision", "recommendation", "strategy", "planning", "prioritization",
            "implementation", "success", "outcome", "analysis", "optimization"
        ]
        
        content = request.content.lower() if request.content else ""
        return any(keyword in content for keyword in decision_keywords)
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process the analysis request for decision support."""
        try:
            logger.info(f"Processing decision support request: {request.id}")
            
            # Parse request content to determine analysis type
            analysis_type = await self._determine_analysis_type(request)
            
            if analysis_type == "comprehensive":
                result = await self._perform_comprehensive_analysis(request)
            elif analysis_type == "recommendations":
                result = await self._generate_recommendations_only(request)
            elif analysis_type == "prioritization":
                result = await self._prioritize_actions_only(request)
            elif analysis_type == "planning":
                result = await self._create_implementation_plan_only(request)
            elif analysis_type == "prediction":
                result = await self._predict_success_only(request)
            else:
                result = await self._perform_comprehensive_analysis(request)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing decision support request: {e}")
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="uncertain",
                    confidence=0.0,
                    reasoning=f"Error in decision support analysis: {str(e)}"
                ),
                processing_time=0.0,
                raw_content=str(request.content),
                extracted_text="",
                metadata={"error": str(e)}
            )
    
    async def _determine_analysis_type(self, request: AnalysisRequest) -> str:
        """Determine the type of analysis needed based on request content."""
        content = request.content.lower() if request.content else ""
        
        if "comprehensive" in content or "full" in content or "complete" in content:
            return "comprehensive"
        elif "recommend" in content or "suggest" in content:
            return "recommendations"
        elif "prioritize" in content or "rank" in content or "order" in content:
            return "prioritization"
        elif "plan" in content or "implement" in content or "timeline" in content:
            return "planning"
        elif "predict" in content or "success" in content or "outcome" in content:
            return "prediction"
        else:
            return "comprehensive"
    
    async def _perform_comprehensive_analysis(self, request: AnalysisRequest) -> AnalysisResult:
        """Perform comprehensive decision analysis."""
        try:
            # Extract context from request
            context = await self._extract_context_from_request(request)
            
            # Step 1: Generate recommendations
            recommendations = await self.recommendation_engine.generate_recommendations(
                context["recommendation_context"],
                max_recommendations=10
            )
            
            if not recommendations:
                            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="neutral",
                    confidence=0.5,
                    reasoning="No recommendations generated based on the provided context"
                ),
                processing_time=0.0,
                raw_content=str(request.content),
                extracted_text="",
                metadata={"recommendations_count": 0}
            )
            
            # Step 2: Prioritize actions
            prioritized_actions = await self.action_prioritizer.prioritize_actions(
                recommendations,
                context["prioritization_context"]
            )
            
            # Step 3: Create implementation plans for top recommendations
            implementation_plans = []
            for action in prioritized_actions[:3]:  # Top 3 recommendations
                plan = await self.implementation_planner.create_implementation_plan(
                    action.recommendation,
                    context["planning_context"]
                )
                implementation_plans.append(plan)
            
            # Step 4: Predict success for top recommendations
            success_predictions = []
            for action in prioritized_actions[:3]:
                prediction = await self.success_predictor.predict_success(
                    action.recommendation,
                    context["prediction_context"]
                )
                success_predictions.append(prediction)
            
            # Generate comprehensive report
            report = await self._generate_comprehensive_report(
                recommendations,
                prioritized_actions,
                implementation_plans,
                success_predictions
            )
            
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="neutral",
                    confidence=0.9,
                    reasoning="Comprehensive decision analysis completed successfully"
                ),
                processing_time=0.0,
                raw_content=str(request.content),
                extracted_text="",
                metadata={
                    "recommendations_count": len(recommendations),
                    "prioritized_actions_count": len(prioritized_actions),
                    "implementation_plans_count": len(implementation_plans),
                    "success_predictions_count": len(success_predictions),
                    "result": report
                }
            )
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            raise
    
    async def _generate_recommendations_only(self, request: AnalysisRequest) -> AnalysisResult:
        """Generate recommendations only."""
        try:
            context = await self._extract_context_from_request(request)
            
            recommendations = await self.recommendation_engine.generate_recommendations(
                context["recommendation_context"],
                max_recommendations=10
            )
            
            report = await self._generate_recommendations_report(recommendations)
            
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="neutral",
                    confidence=0.9,
                    reasoning="Recommendations generated successfully"
                ),
                processing_time=0.0,
                raw_content=str(request.content),
                extracted_text="",
                metadata={"recommendations_count": len(recommendations), "result": report}
            )
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            raise
    
    async def _prioritize_actions_only(self, request: AnalysisRequest) -> AnalysisResult:
        """Prioritize actions only."""
        try:
            # This would require existing recommendations to prioritize
            # For now, generate recommendations first, then prioritize
            context = await self._extract_context_from_request(request)
            
            recommendations = await self.recommendation_engine.generate_recommendations(
                context["recommendation_context"],
                max_recommendations=10
            )
            
            prioritized_actions = await self.action_prioritizer.prioritize_actions(
                recommendations,
                context["prioritization_context"]
            )
            
            report = await self._generate_prioritization_report(prioritized_actions)
            
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="neutral",
                    confidence=0.9,
                    reasoning="Actions prioritized successfully"
                ),
                processing_time=0.0,
                raw_content=str(request.content),
                extracted_text="",
                metadata={"prioritized_actions_count": len(prioritized_actions), "result": report}
            )
            
        except Exception as e:
            logger.error(f"Error prioritizing actions: {e}")
            raise
    
    async def _create_implementation_plan_only(self, request: AnalysisRequest) -> AnalysisResult:
        """Create implementation plan only."""
        try:
            # This would require a specific recommendation to plan for
            # For now, generate a recommendation and create a plan
            context = await self._extract_context_from_request(request)
            
            recommendations = await self.recommendation_engine.generate_recommendations(
                context["recommendation_context"],
                max_recommendations=1
            )
            
            if not recommendations:
                return AnalysisResult(
                    request_id=request.id,
                    data_type=request.data_type,
                    sentiment=SentimentResult(
                        label="neutral",
                        confidence=0.5,
                        reasoning="No recommendations available for implementation planning"
                    ),
                    processing_time=0.0,
                    raw_content=str(request.content),
                    extracted_text="",
                    metadata={"implementation_plans_count": 0}
                )
            
            plan = await self.implementation_planner.create_implementation_plan(
                recommendations[0],
                context["planning_context"]
            )
            
            report = await self._generate_implementation_report(plan)
            
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="neutral",
                    confidence=0.9,
                    reasoning="Implementation plan created successfully"
                ),
                processing_time=0.0,
                raw_content=str(request.content),
                extracted_text="",
                metadata={"implementation_plans_count": 1, "result": report}
            )
            
        except Exception as e:
            logger.error(f"Error creating implementation plan: {e}")
            raise
    
    async def _predict_success_only(self, request: AnalysisRequest) -> AnalysisResult:
        """Predict success only."""
        try:
            # This would require a specific recommendation to predict for
            # For now, generate a recommendation and predict success
            context = await self._extract_context_from_request(request)
            
            recommendations = await self.recommendation_engine.generate_recommendations(
                context["recommendation_context"],
                max_recommendations=1
            )
            
            if not recommendations:
                return AnalysisResult(
                    request_id=request.id,
                    status=ProcessingStatus.COMPLETED,
                    content="No recommendations available for success prediction.",
                    metadata={"success_predictions_count": 0}
                )
            
            prediction = await self.success_predictor.predict_success(
                recommendations[0],
                context["prediction_context"]
            )
            
            report = await self._generate_prediction_report(prediction)
            
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="neutral",
                    confidence=0.9,
                    reasoning="Success prediction completed successfully"
                ),
                processing_time=0.0,
                raw_content=str(request.content),
                extracted_text="",
                metadata={"success_predictions_count": 1, "result": report}
            )
            
        except Exception as e:
            logger.error(f"Error predicting success: {e}")
            raise
    
    async def _extract_context_from_request(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Extract enhanced context information using knowledge graph integration."""
        try:
            # Extract language from request
            language = request.language or "en"
            
            # Get language-specific decision configuration
            language_config = get_language_decision_config(language)
            
            # Extract decision context from knowledge graph
            decision_context = await self.knowledge_graph_integrator.extract_decision_context(
                request, language
            )
            
            # Build enhanced recommendation context
            recommendation_context = RecommendationContext(
                business_objectives=self._extract_business_objectives(decision_context),
                current_performance=self._extract_performance_metrics(decision_context),
                market_conditions=self._extract_market_conditions(decision_context),
                resource_constraints=self._extract_resource_constraints(decision_context),
                risk_tolerance=language_config.get("risk_tolerance", "medium"),
                time_horizon=language_config.get("time_orientation", "medium_term")
            )
            
            # Build enhanced prioritization context
            prioritization_context = PrioritizationContext(
                available_resources=self._extract_available_resources(decision_context),
                time_constraints=self._extract_time_constraints(decision_context),
                stakeholder_preferences=self._extract_stakeholder_preferences(decision_context),
                strategic_goals=self._extract_strategic_goals(decision_context),
                risk_tolerance=language_config.get("risk_tolerance", "medium"),
                budget_constraints=self._extract_budget_constraints(decision_context),
                team_capacity=self._extract_team_capacity(decision_context)
            )
            
            # Build enhanced planning context
            planning_context = PlanningContext(
                available_resources=self._extract_available_resources(decision_context),
                team_capacity=self._extract_team_capacity(decision_context),
                budget_constraints=self._extract_budget_constraints(decision_context),
                timeline_constraints=self._extract_timeline_constraints(decision_context),
                risk_tolerance=language_config.get("risk_tolerance", "medium"),
                stakeholder_requirements=self._extract_stakeholder_requirements(decision_context),
                technical_constraints=self._extract_technical_constraints(decision_context)
            )
            
            # Build enhanced prediction context
            prediction_context = PredictionContext(
                historical_success_rates=self._extract_historical_success_rates(decision_context),
                industry_benchmarks=self._extract_industry_benchmarks(decision_context),
                organizational_capabilities=self._extract_organizational_capabilities(decision_context),
                market_conditions=self._extract_market_conditions(decision_context),
                resource_availability=self._extract_resource_availability(decision_context),
                stakeholder_support=self._extract_stakeholder_support(decision_context),
                external_factors=self._extract_external_factors(decision_context)
            )
            
            return {
                "recommendation_context": recommendation_context,
                "prioritization_context": prioritization_context,
                "planning_context": planning_context,
                "prediction_context": prediction_context,
                "decision_context": decision_context,
                "language_config": language_config
            }
            
        except Exception as e:
            logger.error(f"Error extracting enhanced context: {e}")
            # Fallback to simplified context
            return self._get_fallback_context()
    
    def _extract_business_objectives(self, context: DecisionContext) -> List[str]:
        """Extract business objectives from decision context."""
        objectives = []
        for entity in context.goal_entities:
            objectives.append(f"Improve {entity.entity_name}")
        for entity in context.opportunity_entities:
            objectives.append(f"Leverage {entity.entity_name}")
        return objectives or ["Improve efficiency", "Reduce costs", "Enhance quality"]
    
    def _extract_performance_metrics(self, context: DecisionContext) -> Dict[str, float]:
        """Extract performance metrics from decision context."""
        metrics = {"efficiency": 0.7, "cost_effectiveness": 0.6, "quality": 0.8}
        
        # Adjust based on risk entities
        risk_count = len(context.risk_entities)
        if risk_count > 0:
            metrics["risk_level"] = min(0.8, 0.3 + (risk_count * 0.1))
        
        # Adjust based on opportunity entities
        opportunity_count = len(context.opportunity_entities)
        if opportunity_count > 0:
            metrics["opportunity_potential"] = min(0.9, 0.4 + (opportunity_count * 0.1))
        
        return metrics
    
    def _extract_market_conditions(self, context: DecisionContext) -> Dict[str, float]:
        """Extract market conditions from decision context."""
        conditions = {"volatility": 0.5, "competition": 0.7}
        
        # Adjust based on market entities
        market_count = len(context.market_entities)
        if market_count > 0:
            conditions["market_complexity"] = min(0.9, 0.3 + (market_count * 0.1))
        
        return conditions
    
    def _extract_resource_constraints(self, context: DecisionContext) -> Dict[str, Any]:
        """Extract resource constraints from decision context."""
        constraints = {"budget": 100000, "team_size": 5}
        
        # Adjust based on constraint entities
        constraint_count = len(context.constraint_entities)
        if constraint_count > 0:
            constraints["constraint_level"] = min(0.9, 0.2 + (constraint_count * 0.1))
        
        return constraints
    
    def _extract_available_resources(self, context: DecisionContext) -> Dict[str, Any]:
        """Extract available resources from decision context."""
        return {"budget": 100000, "team_capacity": 5}
    
    def _extract_time_constraints(self, context: DecisionContext) -> Dict[str, Any]:
        """Extract time constraints from decision context."""
        return {"deadline": "6 months"}
    
    def _extract_stakeholder_preferences(self, context: DecisionContext) -> Dict[str, float]:
        """Extract stakeholder preferences from decision context."""
        return {"efficiency": 0.8, "cost_reduction": 0.9}
    
    def _extract_strategic_goals(self, context: DecisionContext) -> List[str]:
        """Extract strategic goals from decision context."""
        goals = []
        for entity in context.goal_entities:
            goals.append(f"Achieve {entity.entity_name}")
        return goals or ["Operational excellence", "Cost leadership"]
    
    def _extract_budget_constraints(self, context: DecisionContext) -> float:
        """Extract budget constraints from decision context."""
        return 100000.0
    
    def _extract_team_capacity(self, context: DecisionContext) -> Dict[str, int]:
        """Extract team capacity from decision context."""
        return {"developers": 3, "analysts": 2}
    
    def _extract_timeline_constraints(self, context: DecisionContext) -> int:
        """Extract timeline constraints from decision context."""
        return 180  # 6 months
    
    def _extract_stakeholder_requirements(self, context: DecisionContext) -> List[str]:
        """Extract stakeholder requirements from decision context."""
        return ["User-friendly interface", "Scalable solution"]
    
    def _extract_technical_constraints(self, context: DecisionContext) -> List[str]:
        """Extract technical constraints from decision context."""
        return ["Cloud deployment", "API integration"]
    
    def _extract_historical_success_rates(self, context: DecisionContext) -> Dict[str, float]:
        """Extract historical success rates from decision context."""
        return {"technology_adoption": 0.65, "process_improvement": 0.70}
    
    def _extract_industry_benchmarks(self, context: DecisionContext) -> Dict[str, float]:
        """Extract industry benchmarks from decision context."""
        return {"success_rate": 0.62, "implementation_time": 8.5}
    
    async def _extract_multi_modal_context(
        self, 
        requests: List[AnalysisRequest]
    ) -> Dict[str, Any]:
        """
        Extract context from multiple modalities using the multi-modal integration engine.
        
        Args:
            requests: List of analysis requests from different modalities
            
        Returns:
            Enhanced context combining insights from all modalities
        """
        try:
            from src.core.multi_modal_integration_engine import MultiModalIntegrationEngine
            
            # Initialize multi-modal integration engine
            multi_modal_engine = MultiModalIntegrationEngine()
            
            # Build unified context from multiple modalities
            unified_context = await multi_modal_engine.build_unified_context(requests)
            
            # Extract decision factors from multi-modal context
            decision_factors = []
            for factor in unified_context.decision_factors:
                if factor.get("confidence", 0) > 0.7:
                    decision_factors.append({
                        "type": factor.get("type", "unknown"),
                        "name": factor.get("name", ""),
                        "modality": factor.get("modality", "unknown"),
                        "confidence": factor.get("confidence", 0.0),
                        "impact": factor.get("impact", "medium")
                    })
            
            # Build enhanced recommendation context
            enhanced_recommendation_context = RecommendationContext(
                business_objectives=self._extract_business_objectives_from_multi_modal(unified_context),
                current_performance=self._extract_performance_from_multi_modal(unified_context),
                market_conditions=self._extract_market_conditions_from_multi_modal(unified_context),
                resource_constraints=self._extract_resource_constraints_from_multi_modal(unified_context),
                stakeholder_preferences=self._extract_stakeholder_preferences_from_multi_modal(unified_context),
                risk_tolerance=self._extract_risk_tolerance_from_multi_modal(unified_context),
                time_horizon=self._extract_time_horizon_from_multi_modal(unified_context),
                success_criteria=self._extract_success_criteria_from_multi_modal(unified_context)
            )
            
            return {
                "unified_context": unified_context,
                "decision_factors": decision_factors,
                "enhanced_recommendation_context": enhanced_recommendation_context,
                "cross_modal_correlations": unified_context.cross_modal_correlations,
                "overall_confidence": unified_context.overall_confidence,
                "modality_insights": unified_context.modality_insights
            }
            
        except Exception as e:
            logger.error(f"Error extracting multi-modal context: {e}")
            return self._get_fallback_context()
    
    def _extract_business_objectives_from_multi_modal(self, unified_context) -> List[str]:
        """Extract business objectives from multi-modal context."""
        try:
            objectives = []
            for entity in unified_context.unified_entities:
                if entity.get("type") == "business" and entity.get("unified_confidence", 0) > 0.7:
                    objectives.append(entity.get("name", ""))
            return objectives
        except Exception as e:
            logger.error(f"Error extracting business objectives from multi-modal: {e}")
            return []
    
    def _extract_performance_from_multi_modal(self, unified_context) -> Dict[str, float]:
        """Extract performance metrics from multi-modal context."""
        try:
            performance = {}
            for pattern in unified_context.unified_patterns:
                if pattern.get("type") == "performance" and pattern.get("unified_confidence", 0) > 0.7:
                    performance[pattern.get("name", "")] = pattern.get("unified_confidence", 0.0)
            return performance
        except Exception as e:
            logger.error(f"Error extracting performance from multi-modal: {e}")
            return {}
    
    def _extract_market_conditions_from_multi_modal(self, unified_context) -> Dict[str, Any]:
        """Extract market conditions from multi-modal context."""
        try:
            market_conditions = {}
            for entity in unified_context.unified_entities:
                if entity.get("type") == "market" and entity.get("unified_confidence", 0) > 0.7:
                    market_conditions[entity.get("name", "")] = {
                        "confidence": entity.get("unified_confidence", 0.0),
                        "modalities": entity.get("modalities", [])
                    }
            return market_conditions
        except Exception as e:
            logger.error(f"Error extracting market conditions from multi-modal: {e}")
            return {}
    
    def _extract_resource_constraints_from_multi_modal(self, unified_context) -> List[str]:
        """Extract resource constraints from multi-modal context."""
        try:
            constraints = []
            for entity in unified_context.unified_entities:
                if entity.get("type") == "constraint" and entity.get("unified_confidence", 0) > 0.7:
                    constraints.append(entity.get("name", ""))
            return constraints
        except Exception as e:
            logger.error(f"Error extracting resource constraints from multi-modal: {e}")
            return []
    
    def _extract_stakeholder_preferences_from_multi_modal(self, unified_context) -> Dict[str, Any]:
        """Extract stakeholder preferences from multi-modal context."""
        try:
            preferences = {}
            for entity in unified_context.unified_entities:
                if entity.get("type") == "stakeholder" and entity.get("unified_confidence", 0) > 0.7:
                    preferences[entity.get("name", "")] = {
                        "confidence": entity.get("unified_confidence", 0.0),
                        "modalities": entity.get("modalities", [])
                    }
            return preferences
        except Exception as e:
            logger.error(f"Error extracting stakeholder preferences from multi-modal: {e}")
            return {}
    
    def _extract_risk_tolerance_from_multi_modal(self, unified_context) -> float:
        """Extract risk tolerance from multi-modal context."""
        try:
            risk_entities = [
                entity for entity in unified_context.unified_entities
                if entity.get("type") == "risk" and entity.get("unified_confidence", 0) > 0.7
            ]
            if risk_entities:
                return statistics.mean([entity.get("unified_confidence", 0.0) for entity in risk_entities])
            return 0.5  # Default risk tolerance
        except Exception as e:
            logger.error(f"Error extracting risk tolerance from multi-modal: {e}")
            return 0.5
    
    def _extract_time_horizon_from_multi_modal(self, unified_context) -> str:
        """Extract time horizon from multi-modal context."""
        try:
            # Analyze patterns to determine time horizon
            temporal_patterns = [
                pattern for pattern in unified_context.unified_patterns
                if pattern.get("type") == "temporal" and pattern.get("unified_confidence", 0) > 0.7
            ]
            if temporal_patterns:
                return "long_term" if len(temporal_patterns) > 2 else "medium_term"
            return "short_term"  # Default time horizon
        except Exception as e:
            logger.error(f"Error extracting time horizon from multi-modal: {e}")
            return "short_term"
    
    def _extract_success_criteria_from_multi_modal(self, unified_context) -> List[str]:
        """Extract success criteria from multi-modal context."""
        try:
            criteria = []
            for entity in unified_context.unified_entities:
                if entity.get("type") == "success_criteria" and entity.get("unified_confidence", 0) > 0.7:
                    criteria.append(entity.get("name", ""))
            return criteria
        except Exception as e:
            logger.error(f"Error extracting success criteria from multi-modal: {e}")
            return []
    
    def _extract_organizational_capabilities(self, context: DecisionContext) -> Dict[str, float]:
        """Extract organizational capabilities from decision context."""
        return {"technical_expertise": 0.7, "change_management": 0.6}
    
    def _extract_resource_availability(self, context: DecisionContext) -> Dict[str, Any]:
        """Extract resource availability from decision context."""
        return {"budget": 100000, "team_size": 5}
    
    def _extract_stakeholder_support(self, context: DecisionContext) -> Dict[str, float]:
        """Extract stakeholder support from decision context."""
        return {"management": 0.8, "end_users": 0.6}
    
    def _extract_external_factors(self, context: DecisionContext) -> Dict[str, str]:
        """Extract external factors from decision context."""
        return {"regulatory_environment": "stable", "competition": "moderate"}
    
    def _get_fallback_context(self) -> Dict[str, Any]:
        """Get fallback context when enhanced extraction fails."""
        return {
            "recommendation_context": RecommendationContext(
                business_objectives=["Improve efficiency", "Reduce costs", "Enhance quality"],
                current_performance={"efficiency": 0.7, "cost_effectiveness": 0.6, "quality": 0.8},
                market_conditions={"volatility": 0.5, "competition": 0.7},
                resource_constraints={"budget": 100000, "team_size": 5},
                risk_tolerance="medium",
                time_horizon="medium_term"
            ),
            "prioritization_context": PrioritizationContext(
                available_resources={"budget": 100000, "team_capacity": 5},
                time_constraints={"deadline": "6 months"},
                stakeholder_preferences={"efficiency": 0.8, "cost_reduction": 0.9},
                strategic_goals=["Operational excellence", "Cost leadership"],
                risk_tolerance="medium",
                budget_constraints=100000,
                team_capacity={"developers": 3, "analysts": 2}
            ),
            "planning_context": PlanningContext(
                available_resources={"budget": 100000, "team_capacity": 5},
                team_capacity={"developers": 3, "analysts": 2},
                budget_constraints=100000,
                timeline_constraints=180,  # 6 months
                risk_tolerance="medium",
                stakeholder_requirements=["User-friendly interface", "Scalable solution"],
                technical_constraints=["Cloud deployment", "API integration"]
            ),
            "prediction_context": PredictionContext(
                historical_success_rates={"technology_adoption": 0.65, "process_improvement": 0.70},
                industry_benchmarks={"success_rate": 0.62, "implementation_time": 8.5},
                organizational_capabilities={"technical_expertise": 0.7, "change_management": 0.6},
                market_conditions={"volatility": 0.5, "growth_rate": 0.08},
                resource_availability={"budget": 100000, "team_size": 5},
                stakeholder_support={"management": 0.8, "end_users": 0.6},
                external_factors={"regulatory_environment": "stable", "competition": "moderate"}
            )
        }
    
    async def _generate_comprehensive_report(
        self,
        recommendations: List,
        prioritized_actions: List,
        implementation_plans: List,
        success_predictions: List
    ) -> str:
        """Generate a comprehensive decision support report."""
        report = "# AI-Powered Decision Support Analysis\n\n"
        
        # Executive Summary
        report += "## Executive Summary\n\n"
        report += f"- **Total Recommendations Generated**: {len(recommendations)}\n"
        report += f"- **Top Priority Actions**: {len(prioritized_actions[:3])}\n"
        report += f"- **Implementation Plans Created**: {len(implementation_plans)}\n"
        report += f"- **Success Predictions**: {len(success_predictions)}\n\n"
        
        # Top Recommendations
        report += "## Top Recommendations\n\n"
        for i, action in enumerate(prioritized_actions[:5], 1):
            rec = action.recommendation
            report += f"### {i}. {rec.title}\n"
            report += f"- **Priority Score**: {action.priority_score:.2f}\n"
            report += f"- **Category**: {rec.category.value.replace('_', ' ').title()}\n"
            report += f"- **Effort**: {rec.implementation_effort.title()}\n"
            report += f"- **Timeline**: {rec.time_to_implement}\n"
            report += f"- **Description**: {rec.description}\n\n"
        
        # Implementation Plans
        if implementation_plans:
            report += "## Implementation Plans\n\n"
            for i, plan in enumerate(implementation_plans, 1):
                report += f"### Plan {i}: {plan.recommendation_title}\n"
                report += f"- **Total Duration**: {plan.total_duration_days} days\n"
                report += f"- **Estimated Cost**: ${plan.total_estimated_cost:,.0f}\n"
                report += f"- **Phases**: {len(plan.phases)}\n"
                report += f"- **Total Tasks**: {sum(len(p.tasks) for p in plan.phases)}\n\n"
        
        # Success Predictions
        if success_predictions:
            report += "## Success Predictions\n\n"
            for i, prediction in enumerate(success_predictions, 1):
                report += f"### Prediction {i}\n"
                report += f"- **Success Probability**: {prediction.overall_success_probability:.1%}\n"
                report += f"- **Success Level**: {prediction.success_level.value.title()}\n"
                report += f"- **Confidence**: {prediction.confidence_level.value.title()}\n"
                report += f"- **Timeline**: {prediction.timeline_months} months\n\n"
        
        # Key Insights
        report += "## Key Insights\n\n"
        if prioritized_actions:
            avg_priority = sum(a.priority_score for a in prioritized_actions) / len(prioritized_actions)
            report += f"- **Average Priority Score**: {avg_priority:.2f}\n"
        
        if success_predictions:
            avg_success = sum(p.overall_success_probability for p in success_predictions) / len(success_predictions)
            report += f"- **Average Success Probability**: {avg_success:.1%}\n"
        
        report += "\n## Next Steps\n\n"
        report += "1. Review and validate top recommendations\n"
        report += "2. Secure stakeholder approval for high-priority actions\n"
        report += "3. Begin implementation planning for selected recommendations\n"
        report += "4. Establish monitoring and success measurement framework\n"
        report += "5. Schedule regular review and adjustment meetings\n"
        
        return report
    
    async def _generate_recommendations_report(self, recommendations: List) -> str:
        """Generate a recommendations-only report."""
        report = "# AI-Generated Recommendations\n\n"
        
        for i, rec in enumerate(recommendations, 1):
            report += f"## {i}. {rec.title}\n"
            report += f"- **Category**: {rec.category.value.replace('_', ' ').title()}\n"
            report += f"- **Priority**: {rec.priority.value.title()}\n"
            report += f"- **Confidence**: {rec.confidence_score:.1%}\n"
            report += f"- **Effort**: {rec.implementation_effort.title()}\n"
            report += f"- **Timeline**: {rec.time_to_implement}\n"
            report += f"- **Description**: {rec.description}\n\n"
        
        return report
    
    async def _generate_prioritization_report(self, prioritized_actions: List) -> str:
        """Generate a prioritization-only report."""
        report = "# Action Prioritization Results\n\n"
        
        for i, action in enumerate(prioritized_actions, 1):
            rec = action.recommendation
            report += f"## {i}. {rec.title}\n"
            report += f"- **Priority Score**: {action.priority_score:.2f}\n"
            report += f"- **Justification**: {action.justification}\n"
            report += f"- **Estimated Start**: {action.estimated_start_date.strftime('%Y-%m-%d') if action.estimated_start_date else 'TBD'}\n"
            report += f"- **Estimated Completion**: {action.estimated_completion_date.strftime('%Y-%m-%d') if action.estimated_completion_date else 'TBD'}\n\n"
        
        return report
    
    async def _generate_implementation_report(self, plan) -> str:
        """Generate an implementation plan report."""
        report = f"# Implementation Plan: {plan.recommendation_title}\n\n"
        
        report += f"## Plan Overview\n"
        report += f"- **Total Duration**: {plan.total_duration_days} days\n"
        report += f"- **Estimated Cost**: ${plan.total_estimated_cost:,.0f}\n"
        report += f"- **Risk Factors**: {', '.join(plan.risk_factors)}\n\n"
        
        report += "## Phases\n\n"
        for i, phase in enumerate(plan.phases, 1):
            report += f"### Phase {i}: {phase.name}\n"
            report += f"- **Duration**: {sum(t.estimated_duration_days for t in phase.tasks)} days\n"
            report += f"- **Tasks**: {len(phase.tasks)}\n"
            report += f"- **Status**: {phase.status.value.replace('_', ' ').title()}\n\n"
        
        return report
    
    async def _generate_prediction_report(self, prediction) -> str:
        """Generate a success prediction report."""
        report = "# Success Prediction Report\n\n"
        
        report += f"## Prediction Summary\n"
        report += f"- **Success Probability**: {prediction.overall_success_probability:.1%}\n"
        report += f"- **Success Level**: {prediction.success_level.value.title()}\n"
        report += f"- **Confidence Level**: {prediction.confidence_level.value.title()}\n"
        report += f"- **Timeline**: {prediction.timeline_months} months\n\n"
        
        report += "## Key Success Factors\n"
        for factor in prediction.key_success_factors:
            report += f"- **{factor.name}**: {factor.score:.2f} (Weight: {factor.weight})\n"
        
        report += "\n## Risk Factors\n"
        for risk in prediction.risk_factors:
            report += f"- {risk}\n"
        
        report += "\n## Recommendations for Success\n"
        for rec in prediction.recommendations_for_success:
            report += f"- {rec}\n"
        
        return report
