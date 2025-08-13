"""
Real-Time Pipeline

Real-time data pipeline for analytics dashboard with:
- ETL/ELT processing
- Data quality management
- Schema evolution
- Data catalog integration
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger

from .data_stream_processor import EnhancedDataStreamProcessor, RealTimeDataPoint
from ...config.real_time_analytics_config import get_real_time_analytics_config


@dataclass
class PipelineStage:
    """Represents a stage in the real-time pipeline."""
    name: str
    processor: Callable
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    timeout: float = 30.0


@dataclass
class PipelineMetrics:
    """Metrics for pipeline performance."""
    total_processed: int = 0
    stage_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    errors: Dict[str, int] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)


class RealTimePipeline:
    """
    Real-time data pipeline for processing and transforming data streams.
    """
    
    def __init__(self, name: str = "default"):
        """Initialize the real-time pipeline."""
        self.name = name
        self.config = get_real_time_analytics_config()
        
        # Pipeline components
        self.stages: List[PipelineStage] = []
        self.stream_processor = EnhancedDataStreamProcessor()
        self.metrics = PipelineMetrics()
        
        # Data quality management
        self.quality_rules: Dict[str, Callable] = {}
        self.schema_validators: Dict[str, Callable] = {}
        
        # Data catalog
        self.data_catalog: Dict[str, Any] = {}
        self.schema_registry: Dict[str, Dict[str, Any]] = {}
        
        # Pipeline state
        self.is_running = False
        self.processing_task = None
        
        logger.info(f"RealTimePipeline '{name}' initialized")
    
    def add_stage(self, name: str, processor: Callable, 
                  config: Optional[Dict[str, Any]] = None) -> None:
        """Add a processing stage to the pipeline."""
        stage = PipelineStage(
            name=name,
            processor=processor,
            config=config or {},
            enabled=True
        )
        self.stages.append(stage)
        self.metrics.stage_metrics[name] = {
            'processed': 0,
            'errors': 0,
            'avg_processing_time': 0.0
        }
        logger.info(f"Added pipeline stage: {name}")
    
    def add_quality_rule(self, name: str, rule: Callable) -> None:
        """Add a data quality rule."""
        self.quality_rules[name] = rule
        logger.info(f"Added quality rule: {name}")
    
    def add_schema_validator(self, name: str, validator: Callable) -> None:
        """Add a schema validator."""
        self.schema_validators[name] = validator
        logger.info(f"Added schema validator: {name}")
    
    async def process_data(self, data: Any, source: str = "default") -> Dict[str, Any]:
        """Process data through the pipeline stages."""
        start_time = datetime.now()
        
        try:
            # Create data point
            data_point = RealTimeDataPoint(
                value=data,
                timestamp=datetime.now(),
                source=source,
                metadata={}
            )
            
            # Process through stages
            result = await self._process_through_stages(data_point)
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.metrics.total_processed += 1
            
            # Update stage metrics
            for stage_name in result.get('stages_processed', []):
                if stage_name in self.metrics.stage_metrics:
                    self.metrics.stage_metrics[stage_name]['processed'] += 1
                    # Update average processing time
                    current_avg = self.metrics.stage_metrics[stage_name]['avg_processing_time']
                    count = self.metrics.stage_metrics[stage_name]['processed']
                    new_avg = (current_avg * (count - 1) + processing_time) / count
                    self.metrics.stage_metrics[stage_name]['avg_processing_time'] = new_avg
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline processing error: {str(e)}")
            self.metrics.errors['pipeline_error'] = self.metrics.errors.get('pipeline_error', 0) + 1
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'source': source
            }
    
    async def _process_through_stages(self, data_point: RealTimeDataPoint) -> Dict[str, Any]:
        """Process data through all pipeline stages."""
        result = {
            'original_data': data_point.value,
            'processed_data': data_point.value,
            'stages_processed': [],
            'quality_scores': {},
            'metadata': data_point.metadata.copy()
        }
        
        current_data = data_point.value
        
        for stage in self.stages:
            if not stage.enabled:
                continue
            
            try:
                stage_start = datetime.now()
                
                # Process through stage
                if asyncio.iscoroutinefunction(stage.processor):
                    processed_data = await asyncio.wait_for(
                        stage.processor(current_data, stage.config),
                        timeout=stage.timeout
                    )
                else:
                    processed_data = stage.processor(current_data, stage.config)
                
                # Update current data
                current_data = processed_data
                result['processed_data'] = processed_data
                result['stages_processed'].append(stage.name)
                
                # Update stage metrics
                stage_time = (datetime.now() - stage_start).total_seconds()
                if stage.name in self.metrics.stage_metrics:
                    self.metrics.stage_metrics[stage.name]['avg_processing_time'] = stage_time
                
                # Apply quality rules
                quality_score = await self._apply_quality_rules(processed_data, stage.name)
                result['quality_scores'][stage.name] = quality_score
                
                # Validate schema
                schema_valid = await self._validate_schema(processed_data, stage.name)
                result['metadata'][f'{stage.name}_schema_valid'] = schema_valid
                
            except asyncio.TimeoutError:
                logger.error(f"Stage {stage.name} timed out")
                self.metrics.stage_metrics[stage.name]['errors'] += 1
                result['metadata'][f'{stage.name}_error'] = 'timeout'
                break
            except Exception as e:
                logger.error(f"Stage {stage.name} error: {str(e)}")
                self.metrics.stage_metrics[stage.name]['errors'] += 1
                result['metadata'][f'{stage.name}_error'] = str(e)
                break
        
        return result
    
    async def _apply_quality_rules(self, data: Any, stage_name: str) -> float:
        """Apply quality rules to data."""
        quality_score = 1.0
        
        for rule_name, rule in self.quality_rules.items():
            try:
                if asyncio.iscoroutinefunction(rule):
                    score = await rule(data, stage_name)
                else:
                    score = rule(data, stage_name)
                quality_score = min(quality_score, score)
            except Exception as e:
                logger.warning(f"Quality rule {rule_name} failed: {str(e)}")
        
        return quality_score
    
    async def _validate_schema(self, data: Any, stage_name: str) -> bool:
        """Validate data schema."""
        for validator_name, validator in self.schema_validators.items():
            try:
                if asyncio.iscoroutinefunction(validator):
                    is_valid = await validator(data, stage_name)
                else:
                    is_valid = validator(data, stage_name)
                
                if not is_valid:
                    return False
            except Exception as e:
                logger.warning(f"Schema validator {validator_name} failed: {str(e)}")
                return False
        
        return True
    
    async def start_pipeline(self) -> None:
        """Start the real-time pipeline."""
        if self.is_running:
            logger.warning("Pipeline is already running")
            return
        
        self.is_running = True
        
        # Start stream processor
        await self.stream_processor.start_enhanced_processing()
        
        # Start pipeline processing task
        self.processing_task = asyncio.create_task(self._pipeline_loop())
        
        logger.info(f"Real-time pipeline '{self.name}' started")
    
    async def stop_pipeline(self) -> None:
        """Stop the real-time pipeline."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop stream processor
        await self.stream_processor.stop_enhanced_processing()
        
        # Stop pipeline processing task
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"Real-time pipeline '{self.name}' stopped")
    
    async def _pipeline_loop(self) -> None:
        """Main pipeline processing loop."""
        while self.is_running:
            try:
                # Process any pending data
                await self._process_pending_data()
                
                # Update data catalog
                await self._update_data_catalog()
                
                # Sleep based on configuration
                await asyncio.sleep(self.config.stream_processing.processing_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pipeline loop error: {str(e)}")
                await asyncio.sleep(1.0)
    
    async def _process_pending_data(self) -> None:
        """Process any pending data in the pipeline."""
        # This would integrate with the stream processor
        # For now, just update metrics
        pass
    
    async def _update_data_catalog(self) -> None:
        """Update the data catalog with current pipeline state."""
        self.data_catalog.update({
            'last_updated': datetime.now().isoformat(),
            'pipeline_name': self.name,
            'total_processed': self.metrics.total_processed,
            'stage_metrics': self.metrics.stage_metrics,
            'errors': self.metrics.errors,
            'uptime': (datetime.now() - self.metrics.start_time).total_seconds()
        })
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get current pipeline metrics."""
        return {
            'pipeline_name': self.name,
            'is_running': self.is_running,
            'total_processed': self.metrics.total_processed,
            'stage_metrics': self.metrics.stage_metrics,
            'errors': self.metrics.errors,
            'uptime': (datetime.now() - self.metrics.start_time).total_seconds(),
            'stream_metrics': self.stream_processor.get_stream_metrics()
        }
    
    def get_data_catalog(self) -> Dict[str, Any]:
        """Get the current data catalog."""
        return self.data_catalog.copy()
    
    def register_schema(self, name: str, schema: Dict[str, Any]) -> None:
        """Register a schema in the schema registry."""
        self.schema_registry[name] = {
            'schema': schema,
            'registered_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        logger.info(f"Registered schema: {name}")


# Factory function for creating real-time pipeline
def create_real_time_pipeline(name: str = "default") -> RealTimePipeline:
    """Create a real-time pipeline with default configuration."""
    return RealTimePipeline(name)
