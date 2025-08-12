"""
Phase 3 Testing Framework for Advanced Features.
Implements automated testing, performance benchmarking, and quality metrics.
"""

import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from src.core.semantic_similarity_analyzer import SemanticSimilarityAnalyzer
from src.core.relationship_optimizer import RelationshipOptimizer
from src.core.chinese_entity_clustering import ChineseEntityClustering


@dataclass
class TestResult:
    """Result of a single test."""
    test_name: str
    status: str  # "PASS", "FAIL", "WARNING"
    execution_time: float
    metrics: Dict[str, Any]
    errors: List[str]
    warnings: List[str]


@dataclass
class BenchmarkResult:
    """Result of performance benchmarking."""
    test_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    throughput: float  # operations per second


class Phase3TestingFramework:
    """Comprehensive testing framework for Phase 3 features."""
    
    def __init__(self):
        self.semantic_analyzer = SemanticSimilarityAnalyzer()
        self.relationship_optimizer = RelationshipOptimizer()
        self.entity_clustering = ChineseEntityClustering()
        
        # Test data for Chinese content
        self.test_data = {
            "chinese_sample": {
                "text": """
                人工智能技术在各个领域都有广泛应用。清华大学和北京大学在机器学习研究方面
                进行了深入合作。北京中关村科技园区聚集了众多高科技企业，包括百度、阿里巴巴
                和腾讯等知名公司。这些企业在人工智能、大数据和云计算技术方面都有重要突破。
                政府相关部门也出台了支持政策，促进技术创新和产业发展。
                """,
                "expected_entities": 15,
                "expected_relationships": 8,
                "expected_orphan_rate": 0.3
            }
        }
        
        # Quality metrics thresholds
        self.quality_thresholds = {
            "orphan_node_rate": 0.3,  # Target < 30% orphan nodes
            "relationship_coverage": 0.5,  # Target > 0.5 relationships per entity
            "average_quality_score": 0.7,  # Target > 0.7 average quality
            "processing_time": 30.0,  # Target < 30 seconds
            "similarity_accuracy": 0.8  # Target > 80% similarity accuracy
        }
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all Phase 3 tests and return comprehensive results."""
        test_results = []
        benchmark_results = []
        
        # Test 1: Semantic Similarity Analysis
        test_result = self._test_semantic_similarity()
        test_results.append(test_result)
        
        # Test 2: Relationship Optimization
        test_result = self._test_relationship_optimization()
        test_results.append(test_result)
        
        # Test 3: Entity Clustering
        test_result = self._test_entity_clustering()
        test_results.append(test_result)
        
        # Test 4: Integration Testing
        test_result = self._test_integration()
        test_results.append(test_result)
        
        # Test 5: Performance Benchmarking
        benchmark_result = self._benchmark_performance()
        benchmark_results.append(benchmark_result)
        
        # Test 6: Quality Metrics
        test_result = self._test_quality_metrics()
        test_results.append(test_result)
        
        # Generate comprehensive report
        return self._generate_comprehensive_report(test_results, benchmark_results)
    
    def _test_semantic_similarity(self) -> TestResult:
        """Test semantic similarity analysis functionality."""
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            # Test data
            test_text = self.test_data["chinese_sample"]["text"]
            test_entities = [
                {"text": "人工智能", "type": "CONCEPT"},
                {"text": "机器学习", "type": "CONCEPT"},
                {"text": "清华大学", "type": "ORGANIZATION"},
                {"text": "北京大学", "type": "ORGANIZATION"},
                {"text": "百度", "type": "ORGANIZATION"}
            ]
            
            # Run semantic similarity analysis
            similarity_results = self.semantic_analyzer.analyze_semantic_similarity(
                test_entities, test_text
            )
            
            # Calculate metrics
            stats = self.semantic_analyzer.get_similarity_statistics(similarity_results)
            high_similarity_pairs = self.semantic_analyzer.filter_high_similarity_pairs(
                similarity_results, threshold=0.6
            )
            
            # Validate results
            if len(similarity_results) == 0:
                errors.append("No similarity results generated")
            
            if stats["average_similarity"] < 0.3:
                warnings.append("Low average similarity score")
            
            if len(high_similarity_pairs) < 2:
                warnings.append("Few high similarity pairs found")
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name="Semantic Similarity Analysis",
                status="PASS" if not errors else "FAIL",
                execution_time=execution_time,
                metrics={
                    "total_pairs": stats["total_pairs"],
                    "average_similarity": stats["average_similarity"],
                    "high_similarity_pairs": len(high_similarity_pairs),
                    "average_confidence": stats["average_confidence"]
                },
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Semantic Similarity Analysis",
                status="FAIL",
                execution_time=execution_time,
                metrics={},
                errors=[f"Exception: {str(e)}"],
                warnings=[]
            )
    
    def _test_relationship_optimization(self) -> TestResult:
        """Test relationship optimization functionality."""
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            # Test data
            test_relationships = [
                {"source": "人工智能", "target": "机器学习", "relationship_type": "RELATED_TO"},
                {"source": "清华大学", "target": "北京大学", "relationship_type": "COLLABORATES_WITH"},
                {"source": "百度", "target": "人工智能", "relationship_type": "IMPLEMENTED_BY"},
                {"source": "人工智能", "target": "机器学习", "relationship_type": "RELATED_TO"}  # Duplicate
            ]
            
            test_entities = [
                {"text": "人工智能", "type": "CONCEPT"},
                {"text": "机器学习", "type": "CONCEPT"},
                {"text": "清华大学", "type": "ORGANIZATION"},
                {"text": "北京大学", "type": "ORGANIZATION"},
                {"text": "百度", "type": "ORGANIZATION"}
            ]
            
            test_text = self.test_data["chinese_sample"]["text"]
            
            # Run relationship optimization
            optimized_relationships = self.relationship_optimizer.optimize_relationships(
                test_relationships, test_entities, test_text
            )
            
            # Calculate optimization statistics
            stats = self.relationship_optimizer.get_optimization_statistics(
                test_relationships, optimized_relationships
            )
            
            # Validate results
            if len(optimized_relationships) == 0:
                errors.append("No optimized relationships generated")
            
            if stats["quality_improvement"] < 0.1:
                warnings.append("Low quality improvement")
            
            if stats["redundancy_reduction"] < 0.1:
                warnings.append("Low redundancy reduction")
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name="Relationship Optimization",
                status="PASS" if not errors else "FAIL",
                execution_time=execution_time,
                metrics={
                    "original_count": stats["original_count"],
                    "optimized_count": stats["optimized_count"],
                    "quality_improvement": stats["quality_improvement"],
                    "redundancy_reduction": stats["redundancy_reduction"],
                    "average_quality_after": stats["average_quality_after"]
                },
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Relationship Optimization",
                status="FAIL",
                execution_time=execution_time,
                metrics={},
                errors=[f"Exception: {str(e)}"],
                warnings=[]
            )
    
    def _test_entity_clustering(self) -> TestResult:
        """Test entity clustering functionality."""
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            # Test data
            test_entities = [
                {"text": "人工智能", "type": "CONCEPT"},
                {"text": "机器学习", "type": "CONCEPT"},
                {"text": "深度学习", "type": "CONCEPT"},
                {"text": "清华大学", "type": "ORGANIZATION"},
                {"text": "北京大学", "type": "ORGANIZATION"},
                {"text": "百度", "type": "ORGANIZATION"},
                {"text": "阿里巴巴", "type": "ORGANIZATION"}
            ]
            
            test_text = self.test_data["chinese_sample"]["text"]
            
            # Run entity clustering
            clusters = self.entity_clustering.cluster_entities(test_entities, test_text)
            
            # Calculate clustering statistics
            stats = self.entity_clustering.get_cluster_statistics(clusters)
            
            # Validate results
            if len(clusters) == 0:
                errors.append("No clusters generated")
            
            if stats["total_relationships_created"] < 5:
                warnings.append("Few relationships created by clustering")
            
            if stats["average_cluster_size"] < 2:
                warnings.append("Small average cluster size")
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name="Entity Clustering",
                status="PASS" if not errors else "FAIL",
                execution_time=execution_time,
                metrics={
                    "total_clusters": stats["total_clusters"],
                    "total_relationships_created": stats["total_relationships_created"],
                    "average_cluster_size": stats["average_cluster_size"],
                    "average_confidence": stats["average_confidence"]
                },
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Entity Clustering",
                status="FAIL",
                execution_time=execution_time,
                metrics={},
                errors=[f"Exception: {str(e)}"],
                warnings=[]
            )
    
    def _test_integration(self) -> TestResult:
        """Test integration of all Phase 3 features."""
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            # Test data
            test_entities = [
                {"text": "人工智能", "type": "CONCEPT"},
                {"text": "机器学习", "type": "CONCEPT"},
                {"text": "清华大学", "type": "ORGANIZATION"},
                {"text": "北京大学", "type": "ORGANIZATION"},
                {"text": "百度", "type": "ORGANIZATION"}
            ]
            
            test_text = self.test_data["chinese_sample"]["text"]
            
            # Step 1: Semantic similarity analysis
            similarity_results = self.semantic_analyzer.analyze_semantic_similarity(
                test_entities, test_text
            )
            
            # Step 2: Generate relationship suggestions
            relationship_suggestions = self.semantic_analyzer.get_relationship_suggestions(
                similarity_results
            )
            
            # Step 3: Entity clustering
            clusters = self.entity_clustering.cluster_entities(test_entities, test_text)
            
            # Step 4: Combine relationships from different sources
            all_relationships = []
            
            # Add suggested relationships
            for suggestion in relationship_suggestions:
                all_relationships.append({
                    "source": suggestion["entity1"],
                    "target": suggestion["entity2"],
                    "relationship_type": suggestion["suggested_relationship"],
                    "confidence": suggestion["confidence"]
                })
            
            # Add clustered relationships
            for cluster in clusters:
                for rel in cluster.relationships:
                    all_relationships.append({
                        "source": rel[0],
                        "target": rel[1],
                        "relationship_type": rel[2],
                        "confidence": cluster.confidence
                    })
            
            # Step 5: Optimize all relationships
            optimized_relationships = self.relationship_optimizer.optimize_relationships(
                all_relationships, test_entities, test_text
            )
            
            # Calculate integration metrics
            total_entities = len(test_entities)
            total_relationships = len(optimized_relationships)
            orphan_nodes = total_entities - len(set(
                rel["source"] for rel in optimized_relationships
            ).union(set(rel["target"] for rel in optimized_relationships)))
            
            orphan_rate = orphan_nodes / total_entities if total_entities > 0 else 1.0
            relationship_coverage = total_relationships / total_entities if total_entities > 0 else 0.0
            
            # Validate integration results
            if orphan_rate > self.quality_thresholds["orphan_node_rate"]:
                warnings.append(f"High orphan node rate: {orphan_rate:.2f}")
            
            if relationship_coverage < self.quality_thresholds["relationship_coverage"]:
                warnings.append(f"Low relationship coverage: {relationship_coverage:.2f}")
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name="Integration Testing",
                status="PASS" if not errors else "FAIL",
                execution_time=execution_time,
                metrics={
                    "total_entities": total_entities,
                    "total_relationships": total_relationships,
                    "orphan_nodes": orphan_nodes,
                    "orphan_rate": orphan_rate,
                    "relationship_coverage": relationship_coverage
                },
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Integration Testing",
                status="FAIL",
                execution_time=execution_time,
                metrics={},
                errors=[f"Exception: {str(e)}"],
                warnings=[]
            )
    
    def _benchmark_performance(self) -> BenchmarkResult:
        """Benchmark performance of Phase 3 features."""
        start_time = time.time()
        
        # Test data
        test_entities = [
            {"text": "人工智能", "type": "CONCEPT"},
            {"text": "机器学习", "type": "CONCEPT"},
            {"text": "深度学习", "type": "CONCEPT"},
            {"text": "神经网络", "type": "CONCEPT"},
            {"text": "清华大学", "type": "ORGANIZATION"},
            {"text": "北京大学", "type": "ORGANIZATION"},
            {"text": "百度", "type": "ORGANIZATION"},
            {"text": "阿里巴巴", "type": "ORGANIZATION"},
            {"text": "腾讯", "type": "ORGANIZATION"}
        ]
        
        test_text = self.test_data["chinese_sample"]["text"] * 10  # Larger text for benchmarking
        
        # Run all Phase 3 operations
        similarity_results = self.semantic_analyzer.analyze_semantic_similarity(
            test_entities, test_text
        )
        
        relationship_suggestions = self.semantic_analyzer.get_relationship_suggestions(
            similarity_results
        )
        
        clusters = self.entity_clustering.cluster_entities(test_entities, test_text)
        
        # Combine and optimize relationships
        all_relationships = []
        for suggestion in relationship_suggestions:
            all_relationships.append({
                "source": suggestion["entity1"],
                "target": suggestion["entity2"],
                "relationship_type": suggestion["suggested_relationship"],
                "confidence": suggestion["confidence"]
            })
        
        for cluster in clusters:
            for rel in cluster.relationships:
                all_relationships.append({
                    "source": rel[0],
                    "target": rel[1],
                    "relationship_type": rel[2],
                    "confidence": cluster.confidence
                })
        
        optimized_relationships = self.relationship_optimizer.optimize_relationships(
            all_relationships, test_entities, test_text
        )
        
        execution_time = time.time() - start_time
        
        # Calculate throughput (operations per second)
        total_operations = (
            len(similarity_results) + 
            len(relationship_suggestions) + 
            len(clusters) + 
            len(optimized_relationships)
        )
        throughput = total_operations / execution_time if execution_time > 0 else 0
        
        return BenchmarkResult(
            test_name="Performance Benchmarking",
            execution_time=execution_time,
            memory_usage=0.0,  # Would need psutil for actual measurement
            cpu_usage=0.0,     # Would need psutil for actual measurement
            throughput=throughput
        )
    
    def _test_quality_metrics(self) -> TestResult:
        """Test quality metrics and thresholds."""
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            # Run integration test to get quality metrics
            integration_result = self._test_integration()
            
            # Check quality thresholds
            metrics = integration_result.metrics
            
            if metrics.get("orphan_rate", 1.0) > self.quality_thresholds["orphan_node_rate"]:
                errors.append(
                    f"Orphan node rate {metrics.get('orphan_rate', 1.0):.2f} exceeds threshold "
                    f"{self.quality_thresholds['orphan_node_rate']}"
                )
            
            if metrics.get("relationship_coverage", 0.0) < self.quality_thresholds["relationship_coverage"]:
                errors.append(
                    f"Relationship coverage {metrics.get('relationship_coverage', 0.0):.2f} below threshold "
                    f"{self.quality_thresholds['relationship_coverage']}"
                )
            
            if integration_result.execution_time > self.quality_thresholds["processing_time"]:
                warnings.append(
                    f"Processing time {integration_result.execution_time:.2f}s exceeds threshold "
                    f"{self.quality_thresholds['processing_time']}s"
                )
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name="Quality Metrics",
                status="PASS" if not errors else "FAIL",
                execution_time=execution_time,
                metrics=metrics,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="Quality Metrics",
                status="FAIL",
                execution_time=execution_time,
                metrics={},
                errors=[f"Exception: {str(e)}"],
                warnings=[]
            )
    
    def _generate_comprehensive_report(
        self, 
        test_results: List[TestResult], 
        benchmark_results: List[BenchmarkResult]
    ) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        # Calculate overall statistics
        total_tests = len(test_results)
        passed_tests = len([r for r in test_results if r.status == "PASS"])
        failed_tests = len([r for r in test_results if r.status == "FAIL"])
        warning_tests = len([r for r in test_results if r.warnings])
        
        total_execution_time = sum(r.execution_time for r in test_results)
        
        # Get benchmark results
        benchmark = benchmark_results[0] if benchmark_results else None
        
        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "phase": "Phase 3 - Advanced Features",
            "overall_status": "PASS" if failed_tests == 0 else "FAIL",
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "warning_tests": warning_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0.0
            },
            "performance_summary": {
                "total_execution_time": total_execution_time,
                "average_execution_time": total_execution_time / total_tests if total_tests > 0 else 0.0,
                "throughput": benchmark.throughput if benchmark else 0.0
            },
            "detailed_results": [
                {
                    "test_name": result.test_name,
                    "status": result.status,
                    "execution_time": result.execution_time,
                    "metrics": result.metrics,
                    "errors": result.errors,
                    "warnings": result.warnings
                }
                for result in test_results
            ],
            "benchmark_results": [
                {
                    "test_name": benchmark.test_name,
                    "execution_time": benchmark.execution_time,
                    "throughput": benchmark.throughput
                }
                for benchmark in benchmark_results
            ],
            "quality_assessment": {
                "meets_orphan_node_target": failed_tests == 0,
                "meets_performance_target": total_execution_time < self.quality_thresholds["processing_time"],
                "overall_quality_score": passed_tests / total_tests if total_tests > 0 else 0.0
            }
        }
        
        return report
    
    def save_test_report(self, report: Dict[str, Any], output_path: str = None) -> str:
        """Save test report to file."""
        if output_path is None:
            output_path = f"phase3_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return output_path
