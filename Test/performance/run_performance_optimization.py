#!/usr/bin/env python3
"""
Comprehensive Performance Optimization Runner
Executes all performance optimization scripts and generates unified reports.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceOptimizationRunner:
    """Comprehensive performance optimization runner."""
    
    def __init__(self):
        self.results = {
            "optimization_start": datetime.now().isoformat(),
            "optimizations": {},
            "summary": {},
            "recommendations": []
        }
        self.optimization_scripts = [
            "load_testing.py",
            "memory_optimization.py", 
            "database_optimization.py",
            "caching_strategy.py"
        ]
    
    def run_load_testing(self) -> Dict[str, Any]:
        """Run load testing optimization."""
        logger.info("Running load testing optimization")
        
        try:
            from load_testing import LoadTester
            load_tester = LoadTester()
            results = load_tester.run_comprehensive_load_test()
            
            logger.info("Load testing completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Load testing failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    def run_memory_optimization(self) -> Dict[str, Any]:
        """Run memory optimization."""
        logger.info("Running memory optimization")
        
        try:
            from memory_optimization import MemoryOptimizer
            optimizer = MemoryOptimizer()
            results = optimizer.run_comprehensive_optimization()
            
            logger.info("Memory optimization completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    def run_database_optimization(self) -> Dict[str, Any]:
        """Run database optimization."""
        logger.info("Running database optimization")
        
        try:
            from database_optimization import DatabaseOptimizer
            optimizer = DatabaseOptimizer()
            results = optimizer.run_comprehensive_optimization()
            
            logger.info("Database optimization completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    def run_caching_strategy(self) -> Dict[str, Any]:
        """Run caching strategy optimization."""
        logger.info("Running caching strategy optimization")
        
        try:
            from caching_strategy import CachingStrategyOptimizer
            optimizer = CachingStrategyOptimizer()
            results = optimizer.run_comprehensive_caching_optimization()
            
            logger.info("Caching strategy optimization completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Caching strategy optimization failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    def run_all_optimizations(self) -> Dict[str, Any]:
        """Run all performance optimizations."""
        logger.info("Starting comprehensive performance optimization")
        
        # Run all optimization scripts
        optimizations = {
            "load_testing": self.run_load_testing(),
            "memory_optimization": self.run_memory_optimization(),
            "database_optimization": self.run_database_optimization(),
            "caching_strategy": self.run_caching_strategy()
        }
        
        # Generate comprehensive summary
        summary = self._generate_comprehensive_summary(optimizations)
        
        self.results["optimizations"] = optimizations
        self.results["summary"] = summary
        
        # Save results
        self._save_comprehensive_results()
        
        return self.results
    
    def _generate_comprehensive_summary(self, optimizations: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary of all optimization results."""
        summary = {
            "overall_performance_score": 0,
            "optimization_status": {},
            "key_improvements": [],
            "critical_issues": [],
            "recommendations": [],
            "performance_metrics": {}
        }
        
        # Analyze each optimization
        for opt_name, opt_results in optimizations.items():
            if "error" in opt_results:
                summary["optimization_status"][opt_name] = "failed"
                summary["critical_issues"].append(f"{opt_name}: {opt_results['error']}")
            else:
                summary["optimization_status"][opt_name] = "completed"
                
                # Extract key metrics from each optimization
                if opt_name == "load_testing":
                    self._analyze_load_testing_results(opt_results, summary)
                elif opt_name == "memory_optimization":
                    self._analyze_memory_optimization_results(opt_results, summary)
                elif opt_name == "database_optimization":
                    self._analyze_database_optimization_results(opt_results, summary)
                elif opt_name == "caching_strategy":
                    self._analyze_caching_strategy_results(opt_results, summary)
        
        # Calculate overall performance score
        completed_optimizations = [
            status for status in summary["optimization_status"].values() 
            if status == "completed"
        ]
        
        if completed_optimizations:
            # Base score from completed optimizations
            base_score = (len(completed_optimizations) / len(optimizations)) * 100
            
            # Adjust score based on key improvements
            improvement_bonus = len(summary["key_improvements"]) * 5
            issue_penalty = len(summary["critical_issues"]) * 10
            
            summary["overall_performance_score"] = max(0, min(100, base_score + improvement_bonus - issue_penalty))
        
        return summary
    
    def _analyze_load_testing_results(self, results: Dict[str, Any], summary: Dict[str, Any]):
        """Analyze load testing results."""
        tests = results.get("tests", {})
        
        # Check API performance
        api_test = tests.get("api_endpoints", {})
        if api_test.get("success_rate", 0) >= 95:
            summary["key_improvements"].append("API load testing: High success rate achieved")
        else:
            summary["recommendations"].append("Improve API performance under load")
        
        # Check database performance
        db_test = tests.get("database_performance", {})
        if db_test.get("success_rate", 0) >= 95:
            summary["key_improvements"].append("Database performance: Good under load")
        else:
            summary["recommendations"].append("Optimize database performance")
        
        # Check concurrent processing
        concurrent_test = tests.get("concurrent_processing", {})
        if concurrent_test.get("success_rate", 0) >= 90:
            summary["key_improvements"].append("Concurrent processing: Good performance")
        else:
            summary["recommendations"].append("Improve concurrent processing capabilities")
        
        # Store performance metrics
        summary["performance_metrics"]["load_testing"] = {
            "api_success_rate": api_test.get("success_rate", 0),
            "db_success_rate": db_test.get("success_rate", 0),
            "concurrent_success_rate": concurrent_test.get("success_rate", 0)
        }
    
    def _analyze_memory_optimization_results(self, results: Dict[str, Any], summary: Dict[str, Any]):
        """Analyze memory optimization results."""
        optimizations = results.get("optimizations", {})
        
        # Check multi-modal optimization
        multi_modal = optimizations.get("multi_modal_optimization", {})
        if multi_modal:
            best_strategy = min(multi_modal.items(), 
                              key=lambda x: x[1].get("memory_increase", float('inf')))
            summary["key_improvements"].append(f"Memory optimization: Best strategy is {best_strategy[0]}")
        
        # Check caching optimization
        caching = optimizations.get("caching_optimization", {})
        if caching:
            best_cache = min(caching.items(), 
                           key=lambda x: x[1].get("memory_used", float('inf')))
            summary["key_improvements"].append(f"Caching optimization: Best strategy is {best_cache[0]}")
        
        # Check ML model optimization
        ml_models = optimizations.get("ml_model_optimization", {})
        if ml_models:
            total_saved = sum(
                strategy.get("memory_saved", 0) for strategy in ml_models.values()
            )
            if total_saved > 0:
                summary["key_improvements"].append(f"ML model optimization: {total_saved / (1024*1024):.1f}MB saved")
        
        # Store performance metrics
        summary["performance_metrics"]["memory_optimization"] = {
            "total_memory_saved_mb": results.get("summary", {}).get("total_memory_saved_mb", 0),
            "optimization_score": results.get("summary", {}).get("optimization_score", 0)
        }
    
    def _analyze_database_optimization_results(self, results: Dict[str, Any], summary: Dict[str, Any]):
        """Analyze database optimization results."""
        optimizations = results.get("optimizations", {})
        
        # Check connection pooling
        pooling = optimizations.get("connection_pooling", {})
        if pooling:
            best_pool = max(pooling.items(), 
                          key=lambda x: x[1].get("requests_per_second", 0))
            summary["key_improvements"].append(f"Database pooling: {best_pool[0]} ({best_pool[1]['requests_per_second']:.1f} req/s)")
        
        # Check index optimization
        indexing = optimizations.get("index_optimization", {})
        if indexing:
            total_improvement = 0
            index_count = 0
            for table_results in indexing.values():
                for query_improvement in table_results.get("query_improvements", {}).values():
                    total_improvement += query_improvement.get("improvement_percent", 0)
                    index_count += 1
            
            if index_count > 0:
                avg_improvement = total_improvement / index_count
                summary["key_improvements"].append(f"Database indexing: {avg_improvement:.1f}% average improvement")
        
        # Check query optimization
        query_opt = optimizations.get("query_optimization", {})
        if query_opt:
            total_query_improvement = 0
            query_count = 0
            for opt_result in query_opt.values():
                total_query_improvement += opt_result.get("improvement_percent", 0)
                query_count += 1
            
            if query_count > 0:
                avg_query_improvement = total_query_improvement / query_count
                summary["key_improvements"].append(f"Query optimization: {avg_query_improvement:.1f}% average improvement")
        
        # Store performance metrics
        summary["performance_metrics"]["database_optimization"] = {
            "overall_performance_score": results.get("summary", {}).get("overall_performance_score", 0),
            "total_optimizations": results.get("summary", {}).get("total_optimizations", 0)
        }
    
    def _analyze_caching_strategy_results(self, results: Dict[str, Any], summary: Dict[str, Any]):
        """Analyze caching strategy results."""
        strategies = results.get("strategies", {})
        
        # Check Redis caching
        redis_results = strategies.get("redis_caching", {})
        if redis_results.get("redis_available"):
            summary["key_improvements"].append("Redis caching: Available and configured")
        else:
            summary["recommendations"].append("Set up Redis for improved caching")
        
        # Check CDN setup
        cdn_results = strategies.get("cdn_setup", {})
        if cdn_results.get("assets_configured", 0) > 0:
            summary["key_improvements"].append(f"CDN setup: {cdn_results['assets_configured']} assets configured")
        else:
            summary["recommendations"].append("Configure CDN for static assets")
        
        # Check API response caching
        api_caching = strategies.get("api_response_caching", {})
        cache_stats = api_caching.get("cache_stats", {})
        hit_rate = cache_stats.get("hit_rate", 0)
        if hit_rate > 50:
            summary["key_improvements"].append(f"API caching: {hit_rate:.1f}% hit rate")
        else:
            summary["recommendations"].append("Optimize API response caching")
        
        # Store performance metrics
        summary["performance_metrics"]["caching_strategy"] = {
            "overall_caching_score": results.get("summary", {}).get("overall_caching_score", 0),
            "total_strategies": results.get("summary", {}).get("total_strategies", 0)
        }
    
    def _save_comprehensive_results(self):
        """Save comprehensive optimization results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_performance_optimization_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Comprehensive optimization results saved to {filename}")
    
    def generate_optimization_report(self) -> str:
        """Generate a human-readable optimization report."""
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("COMPREHENSIVE PERFORMANCE OPTIMIZATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Overall Summary
        summary = self.results.get("summary", {})
        report_lines.append("OVERALL SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Overall Performance Score: {summary.get('overall_performance_score', 0):.1f}/100")
        report_lines.append(f"Optimizations Completed: {len([s for s in summary.get('optimization_status', {}).values() if s == 'completed'])}")
        report_lines.append(f"Optimizations Failed: {len([s for s in summary.get('optimization_status', {}).values() if s == 'failed'])}")
        report_lines.append("")
        
        # Optimization Status
        report_lines.append("OPTIMIZATION STATUS")
        report_lines.append("-" * 40)
        for opt_name, status in summary.get("optimization_status", {}).items():
            status_icon = "✅" if status == "completed" else "❌"
            report_lines.append(f"{status_icon} {opt_name.replace('_', ' ').title()}: {status}")
        report_lines.append("")
        
        # Key Improvements
        if summary.get("key_improvements"):
            report_lines.append("KEY IMPROVEMENTS")
            report_lines.append("-" * 40)
            for improvement in summary["key_improvements"]:
                report_lines.append(f"🏆 {improvement}")
            report_lines.append("")
        
        # Critical Issues
        if summary.get("critical_issues"):
            report_lines.append("CRITICAL ISSUES")
            report_lines.append("-" * 40)
            for issue in summary["critical_issues"]:
                report_lines.append(f"🚨 {issue}")
            report_lines.append("")
        
        # Recommendations
        if summary.get("recommendations"):
            report_lines.append("RECOMMENDATIONS")
            report_lines.append("-" * 40)
            for recommendation in summary["recommendations"]:
                report_lines.append(f"💡 {recommendation}")
            report_lines.append("")
        
        # Performance Metrics
        if summary.get("performance_metrics"):
            report_lines.append("PERFORMANCE METRICS")
            report_lines.append("-" * 40)
            for category, metrics in summary["performance_metrics"].items():
                report_lines.append(f"{category.replace('_', ' ').title()}:")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        report_lines.append(f"  {metric.replace('_', ' ').title()}: {value:.2f}")
                    else:
                        report_lines.append(f"  {metric.replace('_', ' ').title()}: {value}")
                report_lines.append("")
        
        # Footer
        report_lines.append("=" * 80)
        report_lines.append("End of Report")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)


def main():
    """Main function to run comprehensive performance optimization."""
    print("🚀 Starting Comprehensive Performance Optimization")
    print("=" * 60)
    
    # Initialize runner
    runner = PerformanceOptimizationRunner()
    
    try:
        # Run all optimizations
        results = runner.run_all_optimizations()
        
        # Generate and print report
        report = runner.generate_optimization_report()
        print(report)
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"performance_optimization_report_{timestamp}.txt"
        
        with open(report_filename, 'w') as f:
            f.write(report)
        
        print(f"\n📄 Detailed report saved to: {report_filename}")
        print(f"📊 JSON results saved to: comprehensive_performance_optimization_*.json")
        
        # Print final status
        summary = results.get("summary", {})
        overall_score = summary.get("overall_performance_score", 0)
        
        if overall_score >= 80:
            print(f"\n🎉 Performance optimization completed successfully! Score: {overall_score:.1f}/100")
        elif overall_score >= 60:
            print(f"\n⚠️ Performance optimization completed with issues. Score: {overall_score:.1f}/100")
        else:
            print(f"\n❌ Performance optimization needs attention. Score: {overall_score:.1f}/100")
        
    except Exception as e:
        logger.error(f"Comprehensive performance optimization failed: {e}")
        print(f"❌ Comprehensive performance optimization failed: {e}")


if __name__ == "__main__":
    main()
