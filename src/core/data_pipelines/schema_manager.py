"""
Schema Manager

Dynamic schema management with:
- Schema evolution tracking
- Schema validation
- Schema migration
- Schema versioning
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

from loguru import logger

from src.config.big_data_config import get_big_data_config
from src.core.error_handling_service import ErrorHandlingService


@dataclass
class SchemaField:
    """Represents a schema field."""
    name: str
    data_type: str
    required: bool = False
    default_value: Any = None
    description: str = ""


@dataclass
class Schema:
    """Represents a data schema."""
    name: str
    version: str
    fields: List[SchemaField] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    description: str = ""


class SchemaManager:
    """
    Dynamic schema management.
    """
    
    def __init__(self):
        """Initialize schema manager."""
        self.config = get_big_data_config()
        self.error_handler = ErrorHandlingService()
        
        # Schema registry
        self.schemas: Dict[str, Schema] = {}
        self.schema_versions: Dict[str, List[str]] = {}
        
        logger.info("SchemaManager initialized")
    
    async def register_schema(self, schema: Schema) -> bool:
        """Register a new schema."""
        try:
            schema_key = f"{schema.name}_{schema.version}"
            self.schemas[schema_key] = schema
            
            if schema.name not in self.schema_versions:
                self.schema_versions[schema.name] = []
            self.schema_versions[schema.name].append(schema.version)
            
            logger.info(f"Registered schema: {schema_key}")
            return True
            
        except Exception as e:
            await self.error_handler.handle_error(
                "schema_registration_error",
                f"Failed to register schema: {str(e)}",
                error_data={'schema_name': schema.name, 'error': str(e)}
            )
            return False
    
    async def validate_data_against_schema(self, data: List[Dict[str, Any]], 
                                         schema_name: str, 
                                         schema_version: str) -> Dict[str, Any]:
        """Validate data against a specific schema."""
        try:
            schema_key = f"{schema_name}_{schema_version}"
            if schema_key not in self.schemas:
                return {
                    'valid': False,
                    'error': f"Schema {schema_key} not found"
                }
            
            schema = self.schemas[schema_key]
            validation_results = {
                'valid': True,
                'total_records': len(data),
                'valid_records': 0,
                'invalid_records': 0,
                'errors': []
            }
            
            for record_idx, record in enumerate(data):
                record_valid = True
                record_errors = []
                
                # Check required fields
                for field in schema.fields:
                    if field.required and field.name not in record:
                        record_valid = False
                        record_errors.append(f"Missing required field: {field.name}")
                    elif field.name in record:
                        # Validate data type
                        if not self._validate_field_type(record[field.name], field.data_type):
                            record_valid = False
                            record_errors.append(
                                f"Invalid type for field {field.name}: expected {field.data_type}"
                            )
                
                if record_valid:
                    validation_results['valid_records'] += 1
                else:
                    validation_results['invalid_records'] += 1
                    validation_results['errors'].append({
                        'record_index': record_idx,
                        'errors': record_errors
                    })
            
            validation_results['valid'] = validation_results['invalid_records'] == 0
            
            return validation_results
            
        except Exception as e:
            await self.error_handler.handle_error(
                "schema_validation_error",
                f"Failed to validate data against schema: {str(e)}",
                error_data={'schema_name': schema_name, 'error': str(e)}
            )
            return {
                'valid': False,
                'error': str(e)
            }
    
    def _validate_field_type(self, value: Any, expected_type: str) -> bool:
        """Validate field type."""
        try:
            if expected_type == 'string':
                return isinstance(value, str)
            elif expected_type == 'integer':
                return isinstance(value, int)
            elif expected_type == 'float':
                return isinstance(value, (int, float))
            elif expected_type == 'boolean':
                return isinstance(value, bool)
            elif expected_type == 'array':
                return isinstance(value, list)
            elif expected_type == 'object':
                return isinstance(value, dict)
            else:
                return True  # Unknown type, assume valid
        except Exception:
            return False
    
    async def get_schema_versions(self, schema_name: str) -> List[str]:
        """Get all versions of a schema."""
        return self.schema_versions.get(schema_name, [])
    
    async def get_latest_schema(self, schema_name: str) -> Optional[Schema]:
        """Get the latest version of a schema."""
        versions = await self.get_schema_versions(schema_name)
        if versions:
            latest_version = sorted(versions)[-1]
            schema_key = f"{schema_name}_{latest_version}"
            return self.schemas.get(schema_key)
        return None
