"""
Data Synchronizer

Provides real-time data synchronization capabilities including:
- Multi-source data synchronization
- Change detection and propagation
- Conflict resolution
- Data transformation and mapping
- Real-time monitoring and alerting
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from datetime import datetime, timedelta
import threading
from queue import Queue
import uuid

logger = logging.getLogger(__name__)


class SyncDirection(Enum):
    """Synchronization direction"""
    BIDIRECTIONAL = "bidirectional"
    SOURCE_TO_TARGET = "source_to_target"
    TARGET_TO_SOURCE = "target_to_source"


class ConflictResolution(Enum):
    """Conflict resolution strategies"""
    SOURCE_WINS = "source_wins"
    TARGET_WINS = "target_wins"
    MANUAL = "manual"
    MERGE = "merge"
    TIMESTAMP = "timestamp"


@dataclass
class SyncConfig:
    """Configuration for data synchronization"""
    name: str
    source_name: str
    target_name: str
    direction: SyncDirection
    conflict_resolution: ConflictResolution
    sync_interval: int = 60  # seconds
    batch_size: int = 100
    enabled: bool = True
    transform_rules: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    mapping: Dict[str, str] = field(default_factory=dict)


@dataclass
class SyncEvent:
    """Synchronization event"""
    id: str
    timestamp: datetime
    source: str
    target: str
    operation: str
    data: Dict[str, Any]
    status: str
    error: Optional[str] = None


class DataSynchronizer:
    """
    Manages real-time data synchronization between multiple sources
    """
    
    def __init__(self, api_connector=None, db_connector=None):
        """
        Initialize the data synchronizer
        
        Args:
            api_connector: API connector instance
            db_connector: Database connector instance
        """
        self.api_connector = api_connector
        self.db_connector = db_connector
        self.sync_configs = {}
        self.sync_tasks = {}
        self.event_queue = Queue()
        self.event_history = []
        self.running = False
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for the data synchronizer"""
        logging.basicConfig(level=logging.INFO)
        
    def add_sync_config(self, config: SyncConfig):
        """
        Add a new synchronization configuration
        
        Args:
            config: Synchronization configuration
        """
        self.sync_configs[config.name] = config
        logger.info(f"Added sync configuration: {config.name}")
        
    def remove_sync_config(self, name: str):
        """
        Remove a synchronization configuration
        
        Args:
            name: Name of the sync configuration
        """
        if name in self.sync_configs:
            del self.sync_configs[name]
            if name in self.sync_tasks:
                self.sync_tasks[name].cancel()
                del self.sync_tasks[name]
            logger.info(f"Removed sync configuration: {name}")
            
    def _generate_data_hash(self, data: Dict[str, Any]) -> str:
        """
        Generate hash for data comparison
        
        Args:
            data: Data dictionary
            
        Returns:
            Hash string
        """
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
        
    def _transform_data(self, data: Dict[str, Any], 
                       transform_rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform data according to rules
        
        Args:
            data: Source data
            transform_rules: Transformation rules
            
        Returns:
            Transformed data
        """
        transformed = data.copy()
        
        for field, rule in transform_rules.items():
            if field in transformed:
                if rule.get('type') == 'format':
                    # Format transformation
                    format_str = rule.get('format', '')
                    if format_str and isinstance(transformed[field], str):
                        try:
                            # Apply format transformation
                            if format_str == 'uppercase':
                                transformed[field] = transformed[field].upper()
                            elif format_str == 'lowercase':
                                transformed[field] = transformed[field].lower()
                            elif format_str == 'title':
                                transformed[field] = transformed[field].title()
                        except Exception as e:
                            logger.warning(f"Format transformation failed: {e}")
                            
                elif rule.get('type') == 'mapping':
                    # Value mapping
                    mapping = rule.get('mapping', {})
                    if transformed[field] in mapping:
                        transformed[field] = mapping[transformed[field]]
                        
                elif rule.get('type') == 'calculation':
                    # Calculated field
                    formula = rule.get('formula', '')
                    if formula:
                        try:
                            # Simple formula evaluation (be careful with eval)
                            # In production, use a safer expression evaluator
                            context = transformed.copy()
                            result = eval(formula, {"__builtins__": {}}, context)
                            transformed[field] = result
                        except Exception as e:
                            logger.warning(f"Calculation failed: {e}")
                            
        return transformed
        
    def _apply_mapping(self, data: Dict[str, Any], 
                      mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Apply field mapping
        
        Args:
            data: Source data
            mapping: Field mapping dictionary
            
        Returns:
            Mapped data
        """
        mapped = {}
        for source_field, target_field in mapping.items():
            if source_field in data:
                mapped[target_field] = data[source_field]
        return mapped
        
    def _apply_filters(self, data: Dict[str, Any], 
                      filters: Dict[str, Any]) -> bool:
        """
        Apply filters to data
        
        Args:
            data: Data to filter
            filters: Filter conditions
            
        Returns:
            True if data passes filters, False otherwise
        """
        for field, condition in filters.items():
            if field not in data:
                return False
                
            value = data[field]
            
            if isinstance(condition, dict):
                # Complex condition
                if 'min' in condition and value < condition['min']:
                    return False
                if 'max' in condition and value > condition['max']:
                    return False
                if 'values' in condition and value not in condition['values']:
                    return False
                if 'pattern' in condition:
                    import re
                    if not re.match(condition['pattern'], str(value)):
                        return False
            else:
                # Simple equality
                if value != condition:
                    return False
                    
        return True
        
    async def _sync_data(self, config: SyncConfig) -> SyncEvent:
        """
        Perform data synchronization
        
        Args:
            config: Synchronization configuration
            
        Returns:
            Sync event
        """
        event = SyncEvent(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            source=config.source_name,
            target=config.target_name,
            operation="sync",
            data={},
            status="started"
        )
        
        try:
            # Get source data
            source_data = await self._get_source_data(config.source_name)
            
            # Apply filters
            filtered_data = [
                item for item in source_data 
                if self._apply_filters(item, config.filters)
            ]
            
            # Transform data
            transformed_data = []
            for item in filtered_data:
                transformed = self._transform_data(item, config.transform_rules)
                mapped = self._apply_mapping(transformed, config.mapping)
                transformed_data.append(mapped)
                
            # Get target data for comparison
            target_data = await self._get_target_data(config.target_name)
            
            # Detect changes
            changes = self._detect_changes(transformed_data, target_data)
            
            # Apply changes
            if changes:
                await self._apply_changes(config.target_name, changes)
                
            event.data = {
                'source_count': len(source_data),
                'filtered_count': len(filtered_data),
                'changes_count': len(changes)
            }
            event.status = "completed"
            
        except Exception as e:
            event.status = "failed"
            event.error = str(e)
            logger.error(f"Sync failed for {config.name}: {str(e)}")
            
        self.event_history.append(event)
        return event
        
    async def _get_source_data(self, source_name: str) -> List[Dict[str, Any]]:
        """
        Get data from source
        
        Args:
            source_name: Source name
            
        Returns:
            List of data records
        """
        # This is a placeholder - implement based on your data sources
        # Could be API call, database query, file read, etc.
        if self.api_connector:
            try:
                response = await self.api_connector.make_request(
                    source_name, "data"
                )
                return response.get('data', [])
            except Exception as e:
                logger.error(f"Failed to get data from API source {source_name}: {e}")
                
        if self.db_connector:
            try:
                result = await self.db_connector.execute_query(
                    source_name, "SELECT * FROM data"
                )
                return result
            except Exception as e:
                logger.error(f"Failed to get data from DB source {source_name}: {e}")
                
        return []
        
    async def _get_target_data(self, target_name: str) -> List[Dict[str, Any]]:
        """
        Get data from target
        
        Args:
            target_name: Target name
            
        Returns:
            List of data records
        """
        # Similar to _get_source_data but for target
        return await self._get_source_data(target_name)
        
    def _detect_changes(self, source_data: List[Dict[str, Any]], 
                       target_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect changes between source and target data
        
        Args:
            source_data: Source data
            target_data: Target data
            
        Returns:
            List of changes
        """
        changes = []
        
        # Create lookup for target data
        target_lookup = {}
        for item in target_data:
            if 'id' in item:
                target_lookup[item['id']] = item
                
        # Check for new/updated items
        for source_item in source_data:
            source_id = source_item.get('id')
            if source_id:
                if source_id not in target_lookup:
                    # New item
                    changes.append({
                        'operation': 'insert',
                        'data': source_item
                    })
                else:
                    # Check for updates
                    target_item = target_lookup[source_id]
                    source_hash = self._generate_data_hash(source_item)
                    target_hash = self._generate_data_hash(target_item)
                    
                    if source_hash != target_hash:
                        changes.append({
                            'operation': 'update',
                            'id': source_id,
                            'data': source_item
                        })
                        
        # Check for deleted items (if bidirectional)
        source_lookup = {item.get('id'): item for item in source_data if item.get('id')}
        for target_item in target_data:
            target_id = target_item.get('id')
            if target_id and target_id not in source_lookup:
                changes.append({
                    'operation': 'delete',
                    'id': target_id
                })
                
        return changes
        
    async def _apply_changes(self, target_name: str, 
                           changes: List[Dict[str, Any]]):
        """
        Apply changes to target
        
        Args:
            target_name: Target name
            changes: List of changes to apply
        """
        for change in changes:
            operation = change['operation']
            
            if operation == 'insert':
                await self._insert_data(target_name, change['data'])
            elif operation == 'update':
                await self._update_data(target_name, change['id'], change['data'])
            elif operation == 'delete':
                await self._delete_data(target_name, change['id'])
                
    async def _insert_data(self, target_name: str, data: Dict[str, Any]):
        """Insert data into target"""
        if self.db_connector:
            # Generate INSERT query
            fields = list(data.keys())
            placeholders = ', '.join(['?' for _ in fields])
            query = f"INSERT INTO data ({', '.join(fields)}) VALUES ({placeholders})"
            await self.db_connector.execute_query(target_name, query, list(data.values()))
            
    async def _update_data(self, target_name: str, record_id: str, 
                          data: Dict[str, Any]):
        """Update data in target"""
        if self.db_connector:
            # Generate UPDATE query
            set_clause = ', '.join([f"{field} = ?" for field in data.keys()])
            query = f"UPDATE data SET {set_clause} WHERE id = ?"
            params = list(data.values()) + [record_id]
            await self.db_connector.execute_query(target_name, query, params)
            
    async def _delete_data(self, target_name: str, record_id: str):
        """Delete data from target"""
        if self.db_connector:
            query = "DELETE FROM data WHERE id = ?"
            await self.db_connector.execute_query(target_name, query, [record_id])
            
    async def _sync_loop(self, config: SyncConfig):
        """
        Main synchronization loop
        
        Args:
            config: Synchronization configuration
        """
        while self.running and config.enabled:
            try:
                await self._sync_data(config)
                await asyncio.sleep(config.sync_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync loop error for {config.name}: {str(e)}")
                await asyncio.sleep(config.sync_interval)
                
    def start_sync(self, config_name: str):
        """
        Start synchronization for a configuration
        
        Args:
            config_name: Name of the sync configuration
        """
        if config_name not in self.sync_configs:
            raise ValueError(f"Sync configuration '{config_name}' not found")
            
        config = self.sync_configs[config_name]
        
        if config_name in self.sync_tasks:
            logger.warning(f"Sync already running for {config_name}")
            return
            
        self.sync_tasks[config_name] = asyncio.create_task(
            self._sync_loop(config)
        )
        logger.info(f"Started sync for configuration: {config_name}")
        
    def stop_sync(self, config_name: str):
        """
        Stop synchronization for a configuration
        
        Args:
            config_name: Name of the sync configuration
        """
        if config_name in self.sync_tasks:
            self.sync_tasks[config_name].cancel()
            del self.sync_tasks[config_name]
            logger.info(f"Stopped sync for configuration: {config_name}")
            
    def start_all_syncs(self):
        """Start all enabled synchronizations"""
        self.running = True
        for config_name, config in self.sync_configs.items():
            if config.enabled:
                self.start_sync(config_name)
                
    def stop_all_syncs(self):
        """Stop all synchronizations"""
        self.running = False
        for config_name in list(self.sync_tasks.keys()):
            self.stop_sync(config_name)
            
    def get_sync_status(self, config_name: str) -> Dict[str, Any]:
        """
        Get synchronization status
        
        Args:
            config_name: Name of the sync configuration
            
        Returns:
            Status information
        """
        if config_name not in self.sync_configs:
            return {'error': f"Sync configuration '{config_name}' not found"}
            
        config = self.sync_configs[config_name]
        is_running = config_name in self.sync_tasks
        
        return {
            'name': config_name,
            'source': config.source_name,
            'target': config.target_name,
            'direction': config.direction.value,
            'enabled': config.enabled,
            'running': is_running,
            'sync_interval': config.sync_interval,
            'last_event': self.event_history[-1] if self.event_history else None
        }
        
    def get_event_history(self, limit: int = 100) -> List[SyncEvent]:
        """
        Get synchronization event history
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of sync events
        """
        return self.event_history[-limit:] if self.event_history else []
