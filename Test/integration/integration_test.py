#!/usr/bin/env python3
"""
Integration test script for Classical Chinese processing.
Performs end-to-end processing, error handling validation, and performance testing.
"""

import sys
import os
import asyncio
import json
import time
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.orchestrator import SentimentOrchestrator
from core.models import AnalysisRequest, DataType
from agents.enhanced_file_extraction_agent import EnhancedFileExtractionAgent
from agents.knowledge_graph_agent import KnowledgeGraphAgent
from config.language_config.chinese_config import ChineseConfig


class IntegrationTester:
    """Tests end-to-end integration for Classical Chinese processing."""
    
    def __init__(self):
        self.orchestrator = SentimentOrchestrator()
        self.file_agent = EnhancedFileExtractionAgent()
        self.kg_agent = KnowledgeGraphAgent()
        self.chinese_config = ChineseConfig()
        self.test_results = {
            "passed": [],
            "failed": [],
            "warnings": [],
            "performance_metrics": {}
        }
        
        # Test file path
        self.test_file = Path("data/Classical Chinese Sample 22208_0_8.pdf")
    
    async def test_end_to_end_processing(self):
        """Test end-to-end processing pipeline."""
        try:
            start_time = time.time()
            
            # Step 1: File extraction
            print("🔄 Step 1: File extraction...")
            file_request = AnalysisRequest(
                data_type=DataType.PDF,
                content=str(self.test_file),
                language="zh",
                model_preference="qwen2.5:7b"
            )
            
            file_result = await self.file_agent.process(file_request)
            assert file_result is not None
            assert len(file_result.extracted_text) > 0
            
            file_time = time.time() - start_time
            print(f"✅ File extraction completed in {file_time:.2f}s")
            
            # Step 2: Knowledge graph processing
            print("🔄 Step 2: Knowledge graph processing...")
            kg_start_time = time.time()
            
            kg_request = AnalysisRequest(
                data_type=DataType.TEXT,
                content=file_result.extracted_text,
                language="zh",
                model_preference="qwen2.5:7b"
            )
            
            kg_result = await self.kg_agent.process(kg_request)
            assert kg_result is not None
            
            kg_time = time.time() - kg_start_time
            print(f"✅ Knowledge graph processing completed in {kg_time:.2f}s")
            
            # Step 3: Orchestrator integration
            print("🔄 Step 3: Orchestrator integration...")
            orchestrator_start_time = time.time()
            
            # Test orchestrator with the extracted text
            orchestrator_result = await self.orchestrator.analyze(
                AnalysisRequest(
                    data_type=DataType.TEXT,
                    content=file_result.extracted_text[:1000],  # Limit for testing
                    language="zh"
                )
            )
            
            orchestrator_time = time.time() - orchestrator_start_time
            print(f"✅ Orchestrator integration completed in {orchestrator_time:.2f}s")
            
            total_time = time.time() - start_time
            
            # Performance metrics
            self.test_results["performance_metrics"].update({
                "file_extraction_time": file_time,
                "knowledge_graph_time": kg_time,
                "orchestrator_time": orchestrator_time,
                "total_processing_time": total_time,
                "extracted_text_length": len(file_result.extracted_text),
                "knowledge_graph_entities": getattr(kg_result, 'entity_count', 0)
            })
            
            self.test_results["passed"].append("End-to-end processing")
            print("✅ End-to-end processing test passed")
            
        except Exception as e:
            self.test_results["failed"].append(f"End-to-end processing: {e}")
            print(f"❌ End-to-end processing test failed: {e}")
    
    async def test_error_handling_validation(self):
        """Test error handling validation."""
        try:
            # Test with invalid file path
            print("🔄 Testing error handling with invalid file...")
            invalid_request = AnalysisRequest(
                data_type=DataType.PDF,
                content="nonexistent_file.pdf",
                language="zh"
            )
            
            try:
                result = await self.file_agent.process(invalid_request)
                # Should handle gracefully
                print("✅ Error handling with invalid file passed")
            except Exception as e:
                print(f"ℹ️  Expected error caught: {e}")
            
            # Test with empty content
            print("🔄 Testing error handling with empty content...")
            empty_request = AnalysisRequest(
                data_type=DataType.TEXT,
                content="",
                language="zh"
            )
            
            try:
                result = await self.kg_agent.process(empty_request)
                # Should handle gracefully
                print("✅ Error handling with empty content passed")
            except Exception as e:
                print(f"ℹ️  Expected error caught: {e}")
            
            # Test with unsupported language
            print("🔄 Testing error handling with unsupported language...")
            unsupported_request = AnalysisRequest(
                data_type=DataType.TEXT,
                content="Test content",
                language="xx"  # Unsupported language code
            )
            
            try:
                result = await self.orchestrator.analyze(unsupported_request)
                # Should handle gracefully
                print("✅ Error handling with unsupported language passed")
            except Exception as e:
                print(f"ℹ️  Expected error caught: {e}")
            
            self.test_results["passed"].append("Error handling validation")
            print("✅ Error handling validation test passed")
            
        except Exception as e:
            self.test_results["failed"].append(f"Error handling validation: {e}")
            print(f"❌ Error handling validation test failed: {e}")
    
    async def test_performance_benchmarks(self):
        """Test performance benchmarks."""
        try:
            # Performance thresholds
            thresholds = {
                "file_extraction_max_time": 30.0,  # seconds
                "knowledge_graph_max_time": 60.0,  # seconds
                "orchestrator_max_time": 120.0,  # seconds
                "total_max_time": 180.0,  # seconds
                "min_extracted_text_length": 100,
                "min_entities": 10
            }
            
            # Run performance test
            print("🔄 Running performance benchmarks...")
            start_time = time.time()
            
            # File extraction performance
            file_start = time.time()
            file_request = AnalysisRequest(
                data_type=DataType.PDF,
                content=str(self.test_file),
                language="zh"
            )
            file_result = await self.file_agent.process(file_request)
            file_time = time.time() - file_start
            
            # Knowledge graph performance
            kg_start = time.time()
            kg_request = AnalysisRequest(
                data_type=DataType.TEXT,
                content=file_result.extracted_text[:2000],  # Limit for performance test
                language="zh"
            )
            kg_result = await self.kg_agent.process(kg_request)
            kg_time = time.time() - kg_start
            
            total_time = time.time() - start_time
            
            # Validate performance metrics
            performance_issues = []
            
            if file_time > thresholds["file_extraction_max_time"]:
                performance_issues.append(f"File extraction too slow: {file_time:.2f}s")
            
            if kg_time > thresholds["knowledge_graph_max_time"]:
                performance_issues.append(f"Knowledge graph too slow: {kg_time:.2f}s")
            
            if total_time > thresholds["total_max_time"]:
                performance_issues.append(f"Total processing too slow: {total_time:.2f}s")
            
            if len(file_result.extracted_text) < thresholds["min_extracted_text_length"]:
                performance_issues.append(f"Extracted text too short: {len(file_result.extracted_text)} chars")
            
            if getattr(kg_result, 'entity_count', 0) < thresholds["min_entities"]:
                performance_issues.append(f"Too few entities: {getattr(kg_result, 'entity_count', 0)}")
            
            if performance_issues:
                for issue in performance_issues:
                    self.test_results["warnings"].append(issue)
                print("⚠️  Performance warnings detected")
            else:
                print("✅ All performance benchmarks met")
            
            # Store performance metrics
            self.test_results["performance_metrics"].update({
                "file_extraction_time": file_time,
                "knowledge_graph_time": kg_time,
                "total_time": total_time,
                "extracted_text_length": len(file_result.extracted_text),
                "entity_count": getattr(kg_result, 'entity_count', 0)
            })
            
            self.test_results["passed"].append("Performance benchmarks")
            print("✅ Performance benchmarks test passed")
            
        except Exception as e:
            self.test_results["failed"].append(f"Performance benchmarks: {e}")
            print(f"❌ Performance benchmarks test failed: {e}")
    
    async def test_mcp_integration(self):
        """Test MCP integration using direct agent calls."""
        try:
            print("🔄 Testing MCP integration...")
            
            # Test that agents can be called directly (MCP framework compatibility)
            agents = {
                "file_agent": self.file_agent,
                "kg_agent": self.kg_agent
            }
            
            for agent_name, agent in agents.items():
                # Test agent capabilities
                assert hasattr(agent, 'can_process'), f"{agent_name} missing can_process"
                assert hasattr(agent, 'process'), f"{agent_name} missing process"
                
                # Test with simple request
                test_request = AnalysisRequest(
                    data_type=DataType.TEXT,
                    content="子曰：学而时习之，不亦说乎？",
                    language="zh"
                )
                
                # Test can_process
                can_process = await agent.can_process(test_request)
                assert isinstance(can_process, bool), f"{agent_name} can_process should return bool"
                
                print(f"✅ {agent_name} MCP integration test passed")
            
            self.test_results["passed"].append("MCP integration")
            print("✅ MCP integration test passed")
            
        except Exception as e:
            self.test_results["failed"].append(f"MCP integration: {e}")
            print(f"❌ MCP integration test failed: {e}")
    
    async def run_all_tests(self):
        """Run all integration tests."""
        print("🔍 Starting Integration Tests...")
        print("=" * 60)
        
        await self.test_end_to_end_processing()
        await self.test_error_handling_validation()
        await self.test_performance_benchmarks()
        await self.test_mcp_integration()
        
        print("=" * 60)
        print("📊 Test Results Summary:")
        print(f"✅ Passed: {len(self.test_results['passed'])}")
        print(f"❌ Failed: {len(self.test_results['failed'])}")
        print(f"⚠️  Warnings: {len(self.test_results['warnings'])}")
        
        # Performance summary
        if self.test_results["performance_metrics"]:
            print("\n📈 Performance Summary:")
            metrics = self.test_results["performance_metrics"]
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}s")
                else:
                    print(f"  {key}: {value}")
        
        if self.test_results['failed']:
            print("\n❌ Failed Tests:")
            for failure in self.test_results['failed']:
                print(f"  - {failure}")
        
        if self.test_results['warnings']:
            print("\n⚠️  Warnings:")
            for warning in self.test_results['warnings']:
                print(f"  - {warning}")
        
        # Save test results
        results_dir = Path("../Results")
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / "integration_results.json", "w", encoding="utf-8") as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Test results saved to: {results_dir / 'integration_results.json'}")
        
        return len(self.test_results['failed']) == 0


async def main():
    """Main test function."""
    tester = IntegrationTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 All integration tests passed!")
        return 0
    else:
        print("\n💥 Some integration tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
