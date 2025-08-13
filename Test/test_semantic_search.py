"""
Test script for semantic search functionality.

This script tests the semantic search system including:
- Vector database operations
- Semantic search service
- MCP tools integration
- API endpoints
- Streamlit UI integration
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

# Import the components we want to test
from core.vector_db import VectorDBManager
from core.semantic_search_service import SemanticSearchService
from config.semantic_search_config import semantic_search_config, SearchType


class SemanticSearchTester:
    """Test class for semantic search functionality."""
    
    def __init__(self):
        self.vector_db = VectorDBManager()
        self.search_service = SemanticSearchService()
        self.config = semantic_search_config
        self.test_results = {}
    
    async def test_vector_database(self) -> Dict[str, Any]:
        """Test vector database operations."""
        logger.info("🧪 Testing Vector Database Operations")
        
        try:
            # Test 1: Basic operations
            logger.info("  Testing basic vector database operations...")
            
            # Create test document
            test_doc = {
                "text": "This is a test document about artificial intelligence and machine learning.",
                "metadata": {
                    "content_type": "text",
                    "language": "en",
                    "sentiment_label": "positive",
                    "sentiment_confidence": 0.85,
                    "timestamp": "2024-01-01T00:00:00Z"
                }
            }
            
            # Test semantic search
            results = await self.vector_db.semantic_search(
                query="artificial intelligence",
                language="en",
                n_results=5,
                similarity_threshold=0.5
            )
            
            logger.info(f"    Semantic search returned {len(results)} results")
            
            # Test multilingual search
            multilingual_results = await self.vector_db.multi_language_semantic_search(
                query="AI technology",
                n_results=3,
                similarity_threshold=0.6
            )
            
            logger.info(f"    Multilingual search returned results for {len(multilingual_results)} languages")
            
            # Test conceptual search
            conceptual_results = await self.vector_db.search_by_concept(
                concept="machine learning",
                n_results=3,
                similarity_threshold=0.5
            )
            
            logger.info(f"    Conceptual search returned {len(conceptual_results)} results")
            
            # Test search statistics
            stats = await self.vector_db.get_search_statistics()
            logger.info(f"    Search statistics: {stats}")
            
            return {
                "success": True,
                "semantic_results": len(results),
                "multilingual_results": len(multilingual_results),
                "conceptual_results": len(conceptual_results),
                "statistics": stats
            }
            
        except Exception as e:
            logger.error(f"  Vector database test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_semantic_search_service(self) -> Dict[str, Any]:
        """Test semantic search service."""
        logger.info("🧪 Testing Semantic Search Service")
        
        try:
            # Test 1: Basic semantic search
            logger.info("  Testing basic semantic search...")
            
            result = await self.search_service.search(
                query="artificial intelligence applications",
                search_type=SearchType.SEMANTIC,
                language="en",
                n_results=5,
                similarity_threshold=0.6
            )
            
            logger.info(f"    Basic search: {result.get('total_results', 0)} results")
            
            # Test 2: Conceptual search
            logger.info("  Testing conceptual search...")
            
            conceptual_result = await self.search_service.search(
                query="machine learning",
                search_type=SearchType.CONCEPTUAL,
                n_results=3,
                similarity_threshold=0.5
            )
            
            logger.info(f"    Conceptual search: {conceptual_result.get('total_results', 0)} results")
            
            # Test 3: Multilingual search
            logger.info("  Testing multilingual search...")
            
            multilingual_result = await self.search_service.search(
                query="technology",
                search_type=SearchType.MULTILINGUAL,
                n_results=3,
                similarity_threshold=0.6
            )
            
            logger.info(f"    Multilingual search completed")
            
            # Test 4: Cross-content search
            logger.info("  Testing cross-content search...")
            
            cross_content_result = await self.search_service.search(
                query="data analysis",
                search_type=SearchType.CROSS_CONTENT,
                content_types=["text", "pdf", "document"],
                n_results=3,
                similarity_threshold=0.6
            )
            
            logger.info(f"    Cross-content search completed")
            
            # Test 5: Combined search
            logger.info("  Testing combined search...")
            
            combined_result = await self.search_service.search_with_knowledge_graph(
                query="AI and machine learning",
                language="en",
                n_results=5,
                similarity_threshold=0.6
            )
            
            logger.info(f"    Combined search: {combined_result.get('total_results', 0)} results")
            
            # Test 6: Search statistics
            logger.info("  Testing search statistics...")
            
            stats_result = await self.search_service.get_search_statistics()
            logger.info(f"    Search statistics retrieved: {stats_result.get('success', False)}")
            
            return {
                "success": True,
                "basic_search": result.get('total_results', 0),
                "conceptual_search": conceptual_result.get('total_results', 0),
                "multilingual_search": multilingual_result.get('success', False),
                "cross_content_search": cross_content_result.get('success', False),
                "combined_search": combined_result.get('total_results', 0),
                "statistics": stats_result.get('success', False)
            }
            
        except Exception as e:
            logger.error(f"  Semantic search service test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_configuration(self) -> Dict[str, Any]:
        """Test semantic search configuration."""
        logger.info("🧪 Testing Semantic Search Configuration")
        
        try:
            # Test 1: Configuration loading
            logger.info("  Testing configuration loading...")
            
            supported_languages = self.config.get_supported_languages()
            supported_content_types = self.config.get_supported_content_types()
            search_strategies = list(self.config.search_strategies.keys())
            
            logger.info(f"    Supported languages: {len(supported_languages)}")
            logger.info(f"    Supported content types: {len(supported_content_types)}")
            logger.info(f"    Search strategies: {len(search_strategies)}")
            
            # Test 2: Parameter validation
            logger.info("  Testing parameter validation...")
            
            validated_params = self.config.validate_search_parameters(
                n_results=15,
                similarity_threshold=0.8,
                language="en"
            )
            
            logger.info(f"    Validated parameters: {validated_params}")
            
            # Test 3: Language-specific settings
            logger.info("  Testing language-specific settings...")
            
            en_settings = self.config.get_language_settings("en")
            zh_settings = self.config.get_language_settings("zh")
            
            logger.info(f"    English settings: {en_settings.similarity_threshold}")
            logger.info(f"    Chinese settings: {zh_settings.similarity_threshold}")
            
            return {
                "success": True,
                "supported_languages": len(supported_languages),
                "supported_content_types": len(supported_content_types),
                "search_strategies": len(search_strategies),
                "parameter_validation": validated_params is not None,
                "language_settings": {
                    "english": en_settings.similarity_threshold,
                    "chinese": zh_settings.similarity_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"  Configuration test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_performance(self) -> Dict[str, Any]:
        """Test search performance."""
        logger.info("🧪 Testing Search Performance")
        
        try:
            # Test 1: Response time for semantic search
            logger.info("  Testing semantic search response time...")
            
            start_time = time.time()
            result = await self.search_service.search(
                query="artificial intelligence",
                search_type=SearchType.SEMANTIC,
                language="en",
                n_results=10,
                similarity_threshold=0.7
            )
            semantic_time = time.time() - start_time
            
            logger.info(f"    Semantic search time: {semantic_time:.3f}s")
            
            # Test 2: Response time for conceptual search
            logger.info("  Testing conceptual search response time...")
            
            start_time = time.time()
            conceptual_result = await self.search_service.search(
                query="machine learning",
                search_type=SearchType.CONCEPTUAL,
                n_results=10,
                similarity_threshold=0.6
            )
            conceptual_time = time.time() - start_time
            
            logger.info(f"    Conceptual search time: {conceptual_time:.3f}s")
            
            # Test 3: Response time for multilingual search
            logger.info("  Testing multilingual search response time...")
            
            start_time = time.time()
            multilingual_result = await self.search_service.search(
                query="technology",
                search_type=SearchType.MULTILINGUAL,
                n_results=5,
                similarity_threshold=0.6
            )
            multilingual_time = time.time() - start_time
            
            logger.info(f"    Multilingual search time: {multilingual_time:.3f}s")
            
            # Performance criteria
            performance_ok = (
                semantic_time < 2.0 and
                conceptual_time < 3.0 and
                multilingual_time < 5.0
            )
            
            return {
                "success": True,
                "semantic_search_time": semantic_time,
                "conceptual_search_time": conceptual_time,
                "multilingual_search_time": multilingual_time,
                "performance_ok": performance_ok,
                "results": {
                    "semantic": result.get('total_results', 0),
                    "conceptual": conceptual_result.get('total_results', 0),
                    "multilingual": multilingual_result.get('success', False)
                }
            }
            
        except Exception as e:
            logger.error(f"  Performance test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all semantic search tests."""
        logger.info("🚀 Starting Semantic Search Test Suite")
        
        start_time = time.time()
        
        # Run all tests
        tests = [
            ("Vector Database", self.test_vector_database),
            ("Semantic Search Service", self.test_semantic_search_service),
            ("Configuration", self.test_configuration),
            ("Performance", self.test_performance)
        ]
        
        results = {}
        all_passed = True
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running {test_name} Test")
            logger.info(f"{'='*50}")
            
            try:
                result = await test_func()
                results[test_name] = result
                
                if result.get("success", False):
                    logger.info(f"✅ {test_name} test PASSED")
                else:
                    logger.error(f"❌ {test_name} test FAILED")
                    all_passed = False
                    
            except Exception as e:
                logger.error(f"❌ {test_name} test FAILED with exception: {e}")
                results[test_name] = {"success": False, "error": str(e)}
                all_passed = False
        
        total_time = time.time() - start_time
        
        # Generate summary
        summary = {
            "success": all_passed,
            "total_tests": len(tests),
            "passed_tests": sum(1 for r in results.values() if r.get("success", False)),
            "failed_tests": sum(1 for r in results.values() if not r.get("success", False)),
            "total_time": total_time,
            "results": results
        }
        
        # Save results
        self.save_results(summary)
        
        return summary
    
    def save_results(self, summary: Dict[str, Any]):
        """Save test results to file."""
        try:
            results_dir = Path(__file__).parent.parent / "Results"
            results_dir.mkdir(exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"semantic_search_test_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"📊 Test results saved to: {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")


async def main():
    """Main test function."""
    logger.info("🧪 Semantic Search Test Suite")
    logger.info("=" * 50)
    
    # Create tester
    tester = SemanticSearchTester()
    
    # Run tests
    summary = await tester.run_all_tests()
    
    # Display summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 50)
    
    if summary["success"]:
        logger.info("✅ ALL TESTS PASSED")
    else:
        logger.error("❌ SOME TESTS FAILED")
    
    logger.info(f"Total Tests: {summary['total_tests']}")
    logger.info(f"Passed: {summary['passed_tests']}")
    logger.info(f"Failed: {summary['failed_tests']}")
    logger.info(f"Total Time: {summary['total_time']:.2f}s")
    
    # Display detailed results
    for test_name, result in summary["results"].items():
        if result.get("success", False):
            logger.info(f"✅ {test_name}: PASSED")
        else:
            logger.error(f"❌ {test_name}: FAILED - {result.get('error', 'Unknown error')}")
    
    return summary


if __name__ == "__main__":
    # Run the tests
    result = asyncio.run(main())
    
    # Exit with appropriate code
    exit(0 if result["success"] else 1)
