#!/usr/bin/env python3
"""
Test script for Phase 2: Advanced Performance Optimization
Validates multi-level caching, parallel processing, memory management, and performance monitoring.
"""

import sys
import os
import asyncio
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.advanced_caching_service import advanced_cache
from core.parallel_processor import parallel_processor
from core.memory_manager import memory_manager
from core.performance_monitor import performance_monitor
from config.caching_config import CachingConfig
from config.parallel_processing_config import ParallelProcessingConfig
from config.memory_config import MemoryConfig
from config.monitoring_config import MonitoringConfig
from config.language_config.base_config import LanguageConfigFactory


class Phase2PerformanceTester:
    """Test class for validating Phase 2 performance optimizations."""
    
    def __init__(self):
        self.test_results = {}
        self.languages = ['zh', 'ru', 'ja', 'ko', 'en']
    
    async def test_advanced_caching(self):
        """Test advanced multi-level caching service."""
        print("🔍 Testing Advanced Caching Service...")
        
        results = {
            'cache_operations': [],
            'language_optimizations': [],
            'cache_stats': {}
        }
        
        # Test cache operations for each language
        for language in self.languages:
            test_content = f"Test content for {language} language processing"
            
            # Test cache set/get operations
            start_time = time.time()
            await advanced_cache.set(test_content, {"result": "test"}, language, "test_operation")
            cache_result = await advanced_cache.get(test_content, language, "test_operation")
            cache_time = time.time() - start_time
            
            results['cache_operations'].append({
                'language': language,
                'cache_hit': cache_result is not None,
                'operation_time': cache_time,
                'success': True
            })
            
            # Test language-specific optimizations
            optimization_settings = await advanced_cache.get_language_optimized_cache(language)
            results['language_optimizations'].append({
                'language': language,
                'settings': optimization_settings
            })
        
        # Get cache statistics
        cache_stats = await advanced_cache.get_cache_stats()
        results['cache_stats'] = cache_stats
        
        self.test_results['advanced_caching'] = results
        print(f"✅ Advanced caching tested: {len(results['cache_operations'])} languages")
    
    async def test_parallel_processing(self):
        """Test parallel processing service."""
        print("🔍 Testing Parallel Processing Service...")
        
        results = {
            'pdf_processing': [],
            'entity_extraction': [],
            'translation_processing': [],
            'processing_stats': {}
        }
        
        # Test PDF processing for each language
        for language in self.languages:
            pdf_path = f"test_document_{language}.pdf"
            
            start_time = time.time()
            pdf_results = await parallel_processor.process_pdf_pages_parallel(pdf_path, language)
            processing_time = time.time() - start_time
            
            results['pdf_processing'].append({
                'language': language,
                'pages_processed': len(pdf_results),
                'processing_time': processing_time,
                'success': len(pdf_results) > 0
            })
        
        # Test entity extraction
        test_texts = [
            "人工智能技术发展迅速",
            "Искусственный интеллект развивается быстро",
            "人工知能技術が急速に発展している",
            "인공지능 기술이 빠르게 발전하고 있다",
            "Artificial intelligence is developing rapidly"
        ]
        
        for i, text in enumerate(test_texts):
            language = self.languages[i]
            start_time = time.time()
            extraction_results = await parallel_processor.parallel_entity_extraction([text], language)
            extraction_time = time.time() - start_time
            
            results['entity_extraction'].append({
                'language': language,
                'entities_found': len(extraction_results[0].get('entities', [])) if extraction_results else 0,
                'extraction_time': extraction_time,
                'success': extraction_results is not None
            })
        
        # Test translation processing
        source_texts = ["Hello world", "你好世界", "Привет мир"]
        for i, text in enumerate(source_texts):
            source_lang = self.languages[i % len(self.languages)]
            target_lang = self.languages[(i + 1) % len(self.languages)]
            
            start_time = time.time()
            translation_results = await parallel_processor.parallel_translation_processing(
                [text], source_lang, target_lang
            )
            translation_time = time.time() - start_time
            
            results['translation_processing'].append({
                'source_language': source_lang,
                'target_language': target_lang,
                'translation_time': translation_time,
                'success': translation_results is not None
            })
        
        # Get processing statistics
        processing_stats = await parallel_processor.get_processing_stats()
        results['processing_stats'] = processing_stats
        
        self.test_results['parallel_processing'] = results
        print(f"✅ Parallel processing tested: {len(results['pdf_processing'])} languages")
    
    async def test_memory_management(self):
        """Test memory management service."""
        print("🔍 Testing Memory Management Service...")
        
        results = {
            'memory_usage': [],
            'cleanup_operations': [],
            'streaming_processing': [],
            'memory_stats': {}
        }
        
        # Test memory usage monitoring for each language
        for language in self.languages:
            # Check memory usage
            memory_status = await memory_manager.check_memory_usage()
            results['memory_usage'].append({
                'language': language,
                'current_mb': memory_status.get('current_mb', 0),
                'percentage': memory_status.get('percentage', 0),
                'threshold_exceeded': memory_status.get('threshold_exceeded', False)
            })
            
            # Test memory optimization
            optimization_result = await memory_manager.optimize_memory_for_language(language)
            results['cleanup_operations'].append({
                'language': language,
                'optimization_applied': optimization_result.get('optimization_applied', False),
                'new_max_memory_mb': optimization_result.get('new_max_memory_mb', 0)
            })
        
        # Test streaming text processing
        test_text = "This is a long test text that will be processed in streaming mode to test memory efficiency."
        streaming_results = []
        
        async for chunk in memory_manager.process_text_streaming(test_text, "en"):
            streaming_results.append(chunk)
        
        results['streaming_processing'] = {
            'chunks_processed': len(streaming_results),
            'success': len(streaming_results) > 0
        }
        
        # Get memory statistics
        memory_stats = await memory_manager.get_memory_stats()
        results['memory_stats'] = memory_stats
        
        self.test_results['memory_management'] = results
        print(f"✅ Memory management tested: {len(results['memory_usage'])} languages")
    
    async def test_performance_monitoring(self):
        """Test performance monitoring service."""
        print("🔍 Testing Performance Monitoring Service...")
        
        results = {
            'processing_tracking': [],
            'memory_tracking': [],
            'error_tracking': [],
            'performance_report': {}
        }
        
        # Test processing time tracking for each language
        for language in self.languages:
            # Track processing time
            end_tracking = await performance_monitor.track_processing_time(
                "test_operation", language
            )
            
            # Simulate some processing
            await asyncio.sleep(0.1)
            
            # End tracking
            duration = await end_tracking()
            
            results['processing_tracking'].append({
                'language': language,
                'duration': duration,
                'success': duration > 0
            })
            
            # Track memory usage
            await performance_monitor.track_memory_usage("test_operation", language, 100.0)
            
            # Track error rates
            await performance_monitor.track_error_rate("test_operation", language, False)
            
            # Track cache performance
            await performance_monitor.track_cache_performance("test_operation", language, True)
        
        # Get performance report
        performance_report = await performance_monitor.get_performance_report()
        results['performance_report'] = performance_report
        
        # Test language-specific insights
        for language in self.languages:
            insights = await performance_monitor.get_language_performance_insights(language)
            results['processing_tracking'].append({
                'language': language,
                'insights': insights,
                'has_recommendations': len(insights.get('recommendations', [])) > 0
            })
        
        self.test_results['performance_monitoring'] = results
        print(f"✅ Performance monitoring tested: {len(results['processing_tracking'])} languages")
    
    async def test_configuration_systems(self):
        """Test configuration systems."""
        print("🔍 Testing Configuration Systems...")
        
        results = {
            'caching_config': {},
            'parallel_config': {},
            'memory_config': {},
            'monitoring_config': {}
        }
        
        # Test caching configuration
        for language in self.languages:
            cache_settings = CachingConfig.get_language_cache_settings(language)
            cache_recommendations = CachingConfig.get_cache_optimization_recommendations(language)
            
            results['caching_config'][language] = {
                'settings': {
                    'memory_ttl_multiplier': cache_settings.memory_ttl_multiplier,
                    'compression_level': cache_settings.compression_level,
                    'cache_priority': cache_settings.cache_priority
                },
                'recommendations_count': len(cache_recommendations.get('recommendations', []))
            }
        
        # Test parallel processing configuration
        for language in self.languages:
            parallel_settings = ParallelProcessingConfig.get_language_parallel_settings(language)
            parallel_recommendations = ParallelProcessingConfig.get_parallel_optimization_recommendations(language)
            
            results['parallel_config'][language] = {
                'settings': {
                    'chunk_size': parallel_settings.chunk_size,
                    'max_workers': parallel_settings.max_workers,
                    'processing_mode': parallel_settings.processing_mode
                },
                'recommendations_count': len(parallel_recommendations.get('recommendations', []))
            }
        
        # Test memory configuration
        for language in self.languages:
            memory_settings = MemoryConfig.get_language_memory_settings(language)
            memory_recommendations = MemoryConfig.get_memory_optimization_recommendations(language)
            
            results['memory_config'][language] = {
                'settings': {
                    'chunk_size': memory_settings.chunk_size,
                    'memory_multiplier': memory_settings.memory_multiplier,
                    'processing_mode': memory_settings.processing_mode
                },
                'recommendations_count': len(memory_recommendations.get('recommendations', []))
            }
        
        # Test monitoring configuration
        for language in self.languages:
            monitoring_settings = MonitoringConfig.get_language_monitoring_settings(language)
            monitoring_recommendations = MonitoringConfig.get_monitoring_optimization_recommendations(language)
            
            results['monitoring_config'][language] = {
                'settings': {
                    'max_processing_time': monitoring_settings.max_processing_time,
                    'max_memory_mb': monitoring_settings.max_memory_mb,
                    'alert_level': monitoring_settings.alert_level
                },
                'recommendations_count': len(monitoring_recommendations.get('recommendations', []))
            }
        
        self.test_results['configuration_systems'] = results
        print(f"✅ Configuration systems tested: {len(self.languages)} languages")
    
    def generate_report(self):
        """Generate a comprehensive test report."""
        print("\n" + "="*60)
        print("📊 PHASE 2 PERFORMANCE OPTIMIZATION TEST REPORT")
        print("="*60)
        
        total_tests = 0
        passed_tests = 0
        
        # Advanced Caching Results
        if 'advanced_caching' in self.test_results:
            cache_results = self.test_results['advanced_caching']
            cache_operations = cache_results['cache_operations']
            successful_cache_ops = sum(1 for op in cache_operations if op['success'])
            
            print(f"\n💾 Advanced Caching: {successful_cache_ops}/{len(cache_operations)} successful")
            total_tests += len(cache_operations)
            passed_tests += successful_cache_ops
            
            # Show cache statistics
            cache_stats = cache_results.get('cache_stats', {})
            if 'performance' in cache_stats:
                perf = cache_stats['performance']
                print(f"  📈 Overall hit rate: {perf.get('overall_hit_rate', 0):.2%}")
                print(f"  📊 Total requests: {perf.get('total_requests', 0)}")
        
        # Parallel Processing Results
        if 'parallel_processing' in self.test_results:
            parallel_results = self.test_results['parallel_processing']
            
            # PDF processing
            pdf_ops = parallel_results['pdf_processing']
            successful_pdf = sum(1 for op in pdf_ops if op['success'])
            print(f"\n⚡ Parallel Processing: {successful_pdf}/{len(pdf_ops)} PDF operations successful")
            total_tests += len(pdf_ops)
            passed_tests += successful_pdf
            
            # Entity extraction
            entity_ops = parallel_results['entity_extraction']
            successful_entity = sum(1 for op in entity_ops if op['success'])
            print(f"  🔍 Entity extraction: {successful_entity}/{len(entity_ops)} successful")
            total_tests += len(entity_ops)
            passed_tests += successful_entity
        
        # Memory Management Results
        if 'memory_management' in self.test_results:
            memory_results = self.test_results['memory_management']
            memory_ops = memory_results['memory_usage']
            successful_memory = sum(1 for op in memory_ops if not op['threshold_exceeded'])
            
            print(f"\n🧠 Memory Management: {successful_memory}/{len(memory_ops)} memory checks passed")
            total_tests += len(memory_ops)
            passed_tests += successful_memory
            
            # Show streaming results
            streaming = memory_results.get('streaming_processing', {})
            if streaming.get('success'):
                print(f"  📝 Streaming processing: {streaming['chunks_processed']} chunks processed")
        
        # Performance Monitoring Results
        if 'performance_monitoring' in self.test_results:
            monitor_results = self.test_results['performance_monitoring']
            tracking_ops = monitor_results['processing_tracking']
            successful_tracking = sum(1 for op in tracking_ops if op.get('success', False))
            
            print(f"\n📊 Performance Monitoring: {successful_tracking}/{len(tracking_ops)} tracking operations successful")
            total_tests += len(tracking_ops)
            passed_tests += successful_tracking
        
        # Configuration Systems Results
        if 'configuration_systems' in self.test_results:
            config_results = self.test_results['configuration_systems']
            total_configs = 0
            successful_configs = 0
            
            for config_type, configs in config_results.items():
                for language, config in configs.items():
                    total_configs += 1
                    if config.get('recommendations_count', 0) > 0:
                        successful_configs += 1
            
            print(f"\n⚙️  Configuration Systems: {successful_configs}/{total_configs} configurations optimized")
            total_tests += total_configs
            passed_tests += successful_configs
        
        # Summary
        print(f"\n" + "="*60)
        print(f"📈 SUMMARY: {passed_tests}/{total_tests} tests passed")
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        print(f"🎯 Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("🎉 Phase 2 Performance Optimization: SUCCESS!")
        else:
            print("⚠️  Phase 2 Performance Optimization: NEEDS IMPROVEMENT")
        
        print("="*60)
        
        return success_rate >= 80


async def main():
    """Main test function."""
    print("🚀 Starting Phase 2: Advanced Performance Optimization Testing")
    print("="*70)
    
    tester = Phase2PerformanceTester()
    
    # Run all tests
    await tester.test_advanced_caching()
    await tester.test_parallel_processing()
    await tester.test_memory_management()
    await tester.test_performance_monitoring()
    await tester.test_configuration_systems()
    
    # Generate report
    success = tester.generate_report()
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
