"""
Simple test for Phase 3.1 components to debug issues
"""

import asyncio
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.decision_support import (
    RecommendationEngine,
    RecommendationContext
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_recommendation_engine_simple():
    """Simple test of recommendation engine."""
    logger.info("Testing Recommendation Engine (Simple)...")
    
    try:
        # Initialize recommendation engine
        engine = RecommendationEngine()
        
        # Create simple test context
        context = RecommendationContext(
            business_objectives=["Improve efficiency"],
            current_performance={"efficiency": 0.65},
            market_conditions={"volatility": 0.6},
            resource_constraints={"budget": 100000}
        )
        
        # Generate recommendations
        recommendations = await engine.generate_recommendations(context, max_recommendations=2)
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"Recommendation {i}: {rec.title}")
            logger.info(f"  Category: {rec.category.value}")
            logger.info(f"  Priority: {rec.priority.value}")
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error testing recommendation engine: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    # Run the simple test
    recommendations = asyncio.run(test_recommendation_engine_simple())
    
    if recommendations:
        print(f"\n✅ Simple test successful! Generated {len(recommendations)} recommendations")
    else:
        print("\n❌ Simple test failed!")
        sys.exit(1)
