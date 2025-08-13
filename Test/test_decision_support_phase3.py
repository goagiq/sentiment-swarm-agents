"""
Test script for Phase 3.1: Intelligent Recommendation Engine

Tests the decision support components including:
- Recommendation Engine
- Action Prioritizer  
- Implementation Planner
- Success Predictor
- Decision Support Agent
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

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
from src.agents.decision_support_agent import DecisionSupportAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_recommendation_engine():
    """Test the recommendation engine component."""
    logger.info("Testing Recommendation Engine...")
    
    try:
        # Initialize recommendation engine
        engine = RecommendationEngine()
        
        # Create test context
        context = RecommendationContext(
            business_objectives=["Improve operational efficiency", "Reduce costs by 20%", "Enhance customer satisfaction"],
            current_performance={"efficiency": 0.65, "cost_effectiveness": 0.55, "customer_satisfaction": 0.75},
            market_conditions={"volatility": 0.6, "competition": 0.8},
            resource_constraints={"budget": 150000, "team_size": 8},
            risk_tolerance="medium",
            time_horizon="medium_term"
        )
        
        # Generate recommendations
        recommendations = await engine.generate_recommendations(context, max_recommendations=5)
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"Recommendation {i}: {rec.title}")
            logger.info(f"  Category: {rec.category.value}")
            logger.info(f"  Priority: {rec.priority.value}")
            logger.info(f"  Confidence: {rec.confidence_score:.2f}")
            logger.info(f"  Effort: {rec.implementation_effort}")
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error testing recommendation engine: {e}")
        return []


async def test_action_prioritizer(recommendations):
    """Test the action prioritizer component."""
    logger.info("Testing Action Prioritizer...")
    
    try:
        # Initialize action prioritizer
        prioritizer = ActionPrioritizer()
        
        # Create test context
        context = PrioritizationContext(
            available_resources={"budget": 150000, "team_capacity": 8},
            time_constraints={"deadline": "9 months"},
            stakeholder_preferences={"efficiency": 0.9, "cost_reduction": 0.8, "quality": 0.7},
            strategic_goals=["Operational excellence", "Cost leadership", "Customer focus"],
            risk_tolerance="medium",
            budget_constraints=150000,
            team_capacity={"developers": 4, "analysts": 3, "managers": 1}
        )
        
        # Prioritize actions
        prioritized_actions = await prioritizer.prioritize_actions(recommendations, context)
        
        logger.info(f"Prioritized {len(prioritized_actions)} actions")
        
        for i, action in enumerate(prioritized_actions[:3], 1):
            logger.info(f"Priority {i}: {action.recommendation.title}")
            logger.info(f"  Priority Score: {action.priority_score:.2f}")
            logger.info(f"  Justification: {action.justification}")
            logger.info(f"  Dependencies: {len(action.dependencies)}")
        
        # Generate prioritization report
        report = await prioritizer.get_prioritization_report(prioritized_actions)
        logger.info(f"Prioritization Report Summary: {report['summary']}")
        
        return prioritized_actions
        
    except Exception as e:
        logger.error(f"Error testing action prioritizer: {e}")
        return []


async def test_implementation_planner(recommendations):
    """Test the implementation planner component."""
    logger.info("Testing Implementation Planner...")
    
    try:
        # Initialize implementation planner
        planner = ImplementationPlanner()
        
        # Create test context
        context = PlanningContext(
            available_resources={"budget": 150000, "team_capacity": 8},
            team_capacity={"developers": 4, "analysts": 3, "managers": 1},
            budget_constraints=150000,
            timeline_constraints=270,  # 9 months
            risk_tolerance="medium",
            stakeholder_requirements=["User-friendly interface", "Scalable solution", "Integration capabilities"],
            technical_constraints=["Cloud deployment", "API integration", "Security compliance"]
        )
        
        # Create implementation plans for top recommendations
        implementation_plans = []
        for rec in recommendations[:2]:  # Top 2 recommendations
            plan = await planner.create_implementation_plan(rec, context)
            implementation_plans.append(plan)
            
            logger.info(f"Created implementation plan for: {plan.recommendation_title}")
            logger.info(f"  Total Duration: {plan.total_duration_days} days")
            logger.info(f"  Estimated Cost: ${plan.total_estimated_cost:,.0f}")
            logger.info(f"  Phases: {len(plan.phases)}")
            logger.info(f"  Total Tasks: {sum(len(p.tasks) for p in plan.phases)}")
        
        return implementation_plans
        
    except Exception as e:
        logger.error(f"Error testing implementation planner: {e}")
        return []


async def test_success_predictor(recommendations):
    """Test the success predictor component."""
    logger.info("Testing Success Predictor...")
    
    try:
        # Initialize success predictor
        predictor = SuccessPredictor()
        
        # Create test context
        context = PredictionContext(
            historical_success_rates={"technology_adoption": 0.68, "process_improvement": 0.72, "risk_management": 0.75},
            industry_benchmarks={"success_rate": 0.65, "implementation_time": 7.2},
            organizational_capabilities={"technical_expertise": 0.75, "change_management": 0.65, "project_management": 0.70},
            market_conditions={"volatility": 0.55, "growth_rate": 0.12},
            resource_availability={"budget": 150000, "team_size": 8},
            stakeholder_support={"management": 0.85, "end_users": 0.70, "IT": 0.80},
            external_factors={"regulatory_environment": "stable", "competition": "moderate", "technology_trends": "positive"}
        )
        
        # Predict success for recommendations
        success_predictions = []
        for rec in recommendations[:2]:  # Top 2 recommendations
            prediction = await predictor.predict_success(rec, context)
            success_predictions.append(prediction)
            
            logger.info(f"Success prediction for: {rec.title}")
            logger.info(f"  Success Probability: {prediction.overall_success_probability:.1%}")
            logger.info(f"  Success Level: {prediction.success_level.value}")
            logger.info(f"  Confidence: {prediction.confidence_level.value}")
            logger.info(f"  Timeline: {prediction.timeline_months} months")
            logger.info(f"  Key Success Factors: {len(prediction.key_success_factors)}")
            logger.info(f"  Risk Factors: {len(prediction.risk_factors)}")
        
        return success_predictions
        
    except Exception as e:
        logger.error(f"Error testing success predictor: {e}")
        return []


async def test_decision_support_agent():
    """Test the decision support agent."""
    logger.info("Testing Decision Support Agent...")
    
    try:
        # Initialize decision support agent
        agent = DecisionSupportAgent()
        
        # Create test request
        from src.core.models import AnalysisRequest
        
        request = AnalysisRequest(
            request_id="test_decision_support_001",
            content="We need comprehensive decision support analysis for improving our operational efficiency and reducing costs. Please provide recommendations, prioritization, implementation plans, and success predictions.",
            file_paths=[],
            metadata={"test": True}
        )
        
        # Process request
        result = await agent.process(request)
        
        logger.info(f"Decision Support Agent Result:")
        logger.info(f"  Status: {result.status}")
        logger.info(f"  Content Length: {len(result.content)} characters")
        logger.info(f"  Metadata: {result.metadata}")
        
        # Save result to file
        output_file = f"Test/Results/decision_support_test_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result.content)
        
        logger.info(f"Result saved to: {output_file}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error testing decision support agent: {e}")
        return None


async def run_comprehensive_test():
    """Run comprehensive test of all Phase 3.1 components."""
    logger.info("=" * 60)
    logger.info("PHASE 3.1: INTELLIGENT RECOMMENDATION ENGINE TEST")
    logger.info("=" * 60)
    
    try:
        # Test 1: Recommendation Engine
        recommendations = await test_recommendation_engine()
        if not recommendations:
            logger.error("Recommendation engine test failed")
            return False
        
        # Test 2: Action Prioritizer
        prioritized_actions = await test_action_prioritizer(recommendations)
        if not prioritized_actions:
            logger.error("Action prioritizer test failed")
            return False
        
        # Test 3: Implementation Planner
        implementation_plans = await test_implementation_planner(recommendations)
        if not implementation_plans:
            logger.error("Implementation planner test failed")
            return False
        
        # Test 4: Success Predictor
        success_predictions = await test_success_predictor(recommendations)
        if not success_predictions:
            logger.error("Success predictor test failed")
            return False
        
        # Test 5: Decision Support Agent
        agent_result = await test_decision_support_agent()
        if not agent_result:
            logger.error("Decision support agent test failed")
            return False
        
        # Summary
        logger.info("=" * 60)
        logger.info("PHASE 3.1 TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Recommendation Engine: {len(recommendations)} recommendations generated")
        logger.info(f"✅ Action Prioritizer: {len(prioritized_actions)} actions prioritized")
        logger.info(f"✅ Implementation Planner: {len(implementation_plans)} plans created")
        logger.info(f"✅ Success Predictor: {len(success_predictions)} predictions made")
        logger.info(f"✅ Decision Support Agent: {agent_result.status} status")
        logger.info("=" * 60)
        logger.info("🎉 PHASE 3.1: INTELLIGENT RECOMMENDATION ENGINE - COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Comprehensive test failed: {e}")
        return False


if __name__ == "__main__":
    # Run the comprehensive test
    success = asyncio.run(run_comprehensive_test())
    
    if success:
        print("\n🎉 Phase 3.1 implementation completed successfully!")
        print("Next: Proceed to Phase 3.2 - Risk Assessment & Management")
    else:
        print("\n❌ Phase 3.1 implementation failed!")
        sys.exit(1)
