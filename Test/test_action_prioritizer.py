"""
Test specifically for Action Prioritizer to debug issues
"""

import asyncio
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.decision_support import (
    RecommendationEngine,
    ActionPrioritizer,
    RecommendationContext,
    PrioritizationContext
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_action_prioritizer():
    """Test the action prioritizer component."""
    logger.info("Testing Action Prioritizer...")
    
    try:
        # First, generate some recommendations
        engine = RecommendationEngine()
        context = RecommendationContext(
            business_objectives=["Improve efficiency"],
            current_performance={"efficiency": 0.65},
            market_conditions={"volatility": 0.6},
            resource_constraints={"budget": 100000}
        )
        
        recommendations = await engine.generate_recommendations(context, max_recommendations=2)
        logger.info(f"Generated {len(recommendations)} recommendations")
        
        # Now test action prioritizer
        prioritizer = ActionPrioritizer()
        
        prioritization_context = PrioritizationContext(
            available_resources={"budget": 100000, "team_capacity": 5},
            time_constraints={"deadline": "6 months"},
            stakeholder_preferences={"efficiency": 0.8},
            strategic_goals=["Operational excellence"],
            risk_tolerance="medium",
            budget_constraints=100000,
            team_capacity={"developers": 3, "analysts": 2}
        )
        
        # Prioritize actions
        prioritized_actions = await prioritizer.prioritize_actions(recommendations, prioritization_context)
        
        logger.info(f"Prioritized {len(prioritized_actions)} actions")
        
        for i, action in enumerate(prioritized_actions, 1):
            logger.info(f"Priority {i}: {action.recommendation.title}")
            logger.info(f"  Priority Score: {action.priority_score:.2f}")
            logger.info(f"  Justification: {action.justification}")
        
        return prioritized_actions
        
    except Exception as e:
        logger.error(f"Error testing action prioritizer: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    # Run the test
    actions = asyncio.run(test_action_prioritizer())
    
    if actions:
        print(f"\n✅ Action prioritizer test successful! Prioritized {len(actions)} actions")
    else:
        print("\n❌ Action prioritizer test failed!")
        sys.exit(1)
