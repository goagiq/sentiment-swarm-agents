"""
Data Governance

Data governance and quality management with:
- Data lineage tracking
- Quality rules and validation
- Metadata management
- Compliance monitoring
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib
from pathlib import Path

from loguru import logger

from src.config.big_data_config import get_big_data_config
from src.core.error_handling_service import ErrorHandlingService


@dataclass
class DataLineage:
    """Represents data lineage information."""
    source_id: str
    source_path: str
    transformation_steps: List[str] = field(default_factory=list)
    output_path: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityRule:
    """Represents a data quality rule."""
    name: str
    rule_type: str  # 'validation', 'completeness', 'consistency', 'accuracy'
    rule_function: Callable
    severity: str = 'warning'  # 'error', 'warning', 'info'
    description: str = ""


@dataclass
class QualityReport:
    """Represents a data quality report."""
    dataset_id: str
    timestamp: datetime
    total_records: int
    passed_records: int
    failed_records: int
    quality_score: float
    rule_violations: Dict[str, int] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


class DataGovernance:
    """
    Data governance and quality management.
    """
    
    def __init__(self):
        """Initialize data governance."""
        self.config = get_big_data_config()
        self.error_handler = ErrorHandlingService()
        
        # Data lineage tracking
        self.lineage_registry: Dict[str, DataLineage] = {}
        self.lineage_graph: Dict[str, List[str]] = {}
        
        # Quality management
        self.quality_rules: Dict[str, QualityRule] = {}
        self.quality_reports: Dict[str, QualityReport] = {}
        
        # Metadata management
        self.metadata_registry: Dict[str, Dict[str, Any]] = {}
        
        # Compliance monitoring
        self.compliance_rules: Dict[str, Dict[str, Any]] = {}
        self.compliance_reports: Dict[str, Dict[str, Any]] = {}
        
        logger.info("DataGovernance initialized")
    
    async def register_data_lineage(self, lineage: DataLineage) -> str:
        """Register data lineage information."""
        try:
            # Generate unique ID
            lineage_id = hashlib.md5(
                f"{lineage.source_id}_{lineage.created_at.isoformat()}".encode()
            ).hexdigest()
            
            # Store lineage information
            self.lineage_registry[lineage_id] = lineage
            
            # Update lineage graph
            if lineage.source_id not in self.lineage_graph:
                self.lineage_graph[lineage.source_id] = []
            self.lineage_graph[lineage.source_id].append(lineage_id)
            
            logger.info(f"Registered data lineage: {lineage_id}")
            return lineage_id
            
        except Exception as e:
            await self.error_handler.handle_error(
                "lineage_registration_error",
                f"Failed to register data lineage: {str(e)}",
                error_data={'source_id': lineage.source_id, 'error': str(e)}
            )
            return ""
    
    async def get_data_lineage(self, source_id: str) -> List[DataLineage]:
        """Get lineage information for a data source."""
        try:
            lineage_records = []
            
            # Find all lineage records for the source
            for lineage_id, lineage in self.lineage_registry.items():
                if lineage.source_id == source_id:
                    lineage_records.append(lineage)
            
            return lineage_records
            
        except Exception as e:
            await self.error_handler.handle_error(
                "lineage_retrieval_error",
                f"Failed to get data lineage: {str(e)}",
                error_data={'source_id': source_id, 'error': str(e)}
            )
            return []
    
    async def add_quality_rule(self, rule: QualityRule) -> bool:
        """Add a data quality rule."""
        try:
            self.quality_rules[rule.name] = rule
            logger.info(f"Added quality rule: {rule.name}")
            return True
            
        except Exception as e:
            await self.error_handler.handle_error(
                "quality_rule_error",
                f"Failed to add quality rule: {str(e)}",
                error_data={'rule_name': rule.name, 'error': str(e)}
            )
            return False
    
    async def validate_data_quality(self, dataset_id: str, 
                                  data: List[Dict[str, Any]]) -> QualityReport:
        """Validate data quality using registered rules."""
        try:
            total_records = len(data)
            passed_records = 0
            failed_records = 0
            rule_violations = {}
            
            # Apply each quality rule
            for rule_name, rule in self.quality_rules.items():
                violations = 0
                
                for record in data:
                    try:
                        # Apply rule function
                        if rule.rule_function(record):
                            passed_records += 1
                        else:
                            failed_records += 1
                            violations += 1
                            
                    except Exception as e:
                        logger.warning(
                            f"Rule {rule_name} failed for record: {str(e)}"
                        )
                        failed_records += 1
                        violations += 1
                
                rule_violations[rule_name] = violations
            
            # Calculate quality score
            quality_score = (passed_records / total_records * 100) if total_records > 0 else 0
            
            # Create quality report
            report = QualityReport(
                dataset_id=dataset_id,
                timestamp=datetime.now(),
                total_records=total_records,
                passed_records=passed_records,
                failed_records=failed_records,
                quality_score=quality_score,
                rule_violations=rule_violations
            )
            
            # Store report
            self.quality_reports[dataset_id] = report
            
            logger.info(
                f"Data quality validation completed for {dataset_id}: "
                f"{quality_score:.2f}% quality score"
            )
            
            return report
            
        except Exception as e:
            await self.error_handler.handle_error(
                "quality_validation_error",
                f"Failed to validate data quality: {str(e)}",
                error_data={'dataset_id': dataset_id, 'error': str(e)}
            )
            
            # Return empty report on error
            return QualityReport(
                dataset_id=dataset_id,
                timestamp=datetime.now(),
                total_records=0,
                passed_records=0,
                failed_records=0,
                quality_score=0.0
            )
    
    async def add_metadata(self, dataset_id: str, metadata: Dict[str, Any]) -> bool:
        """Add metadata for a dataset."""
        try:
            if dataset_id not in self.metadata_registry:
                self.metadata_registry[dataset_id] = {}
            
            # Update metadata
            self.metadata_registry[dataset_id].update(metadata)
            self.metadata_registry[dataset_id]['last_updated'] = datetime.now().isoformat()
            
            logger.info(f"Added metadata for dataset: {dataset_id}")
            return True
            
        except Exception as e:
            await self.error_handler.handle_error(
                "metadata_error",
                f"Failed to add metadata: {str(e)}",
                error_data={'dataset_id': dataset_id, 'error': str(e)}
            )
            return False
    
    async def get_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Get metadata for a dataset."""
        try:
            return self.metadata_registry.get(dataset_id, {})
            
        except Exception as e:
            await self.error_handler.handle_error(
                "metadata_retrieval_error",
                f"Failed to get metadata: {str(e)}",
                error_data={'dataset_id': dataset_id, 'error': str(e)}
            )
            return {}
    
    async def add_compliance_rule(self, rule_name: str, 
                                rule_config: Dict[str, Any]) -> bool:
        """Add a compliance rule."""
        try:
            self.compliance_rules[rule_name] = rule_config
            logger.info(f"Added compliance rule: {rule_name}")
            return True
            
        except Exception as e:
            await self.error_handler.handle_error(
                "compliance_rule_error",
                f"Failed to add compliance rule: {str(e)}",
                error_data={'rule_name': rule_name, 'error': str(e)}
            )
            return False
    
    async def check_compliance(self, dataset_id: str, 
                             data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check compliance for a dataset."""
        try:
            compliance_results = {
                'dataset_id': dataset_id,
                'timestamp': datetime.now().isoformat(),
                'rules_checked': len(self.compliance_rules),
                'compliant': True,
                'violations': []
            }
            
            # Check each compliance rule
            for rule_name, rule_config in self.compliance_rules.items():
                try:
                    # Apply compliance check based on rule type
                    if rule_config.get('type') == 'data_retention':
                        # Check data retention compliance
                        retention_days = rule_config.get('retention_days', 365)
                        # Implementation would check data age
                        pass
                    
                    elif rule_config.get('type') == 'data_privacy':
                        # Check data privacy compliance
                        sensitive_fields = rule_config.get('sensitive_fields', [])
                        for record in data:
                            for field in sensitive_fields:
                                if field in record and record[field]:
                                    compliance_results['violations'].append({
                                        'rule': rule_name,
                                        'type': 'sensitive_data_exposure',
                                        'field': field
                                    })
                                    compliance_results['compliant'] = False
                    
                    elif rule_config.get('type') == 'data_format':
                        # Check data format compliance
                        required_fields = rule_config.get('required_fields', [])
                        for record in data:
                            for field in required_fields:
                                if field not in record:
                                    compliance_results['violations'].append({
                                        'rule': rule_name,
                                        'type': 'missing_required_field',
                                        'field': field
                                    })
                                    compliance_results['compliant'] = False
                    
                except Exception as e:
                    logger.warning(f"Compliance rule {rule_name} failed: {str(e)}")
                    compliance_results['violations'].append({
                        'rule': rule_name,
                        'type': 'rule_execution_error',
                        'error': str(e)
                    })
                    compliance_results['compliant'] = False
            
            # Store compliance report
            self.compliance_reports[dataset_id] = compliance_results
            
            logger.info(
                f"Compliance check completed for {dataset_id}: "
                f"{'Compliant' if compliance_results['compliant'] else 'Non-compliant'}"
            )
            
            return compliance_results
            
        except Exception as e:
            await self.error_handler.handle_error(
                "compliance_check_error",
                f"Failed to check compliance: {str(e)}",
                error_data={'dataset_id': dataset_id, 'error': str(e)}
            )
            return {
                'dataset_id': dataset_id,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'compliant': False
            }
    
    async def get_governance_summary(self) -> Dict[str, Any]:
        """Get governance summary statistics."""
        try:
            return {
                'lineage_records': len(self.lineage_registry),
                'quality_rules': len(self.quality_rules),
                'quality_reports': len(self.quality_reports),
                'metadata_datasets': len(self.metadata_registry),
                'compliance_rules': len(self.compliance_rules),
                'compliance_reports': len(self.compliance_reports),
                'total_datasets_tracked': len(set(
                    list(self.metadata_registry.keys()) +
                    list(self.quality_reports.keys()) +
                    list(self.compliance_reports.keys())
                ))
            }
            
        except Exception as e:
            await self.error_handler.handle_error(
                "governance_summary_error",
                f"Failed to get governance summary: {str(e)}",
                error_data={'error': str(e)}
            )
            return {}
    
    async def export_governance_data(self, export_path: str) -> bool:
        """Export governance data to file."""
        try:
            export_data = {
                'lineage_registry': {
                    k: {
                        'source_id': v.source_id,
                        'source_path': v.source_path,
                        'transformation_steps': v.transformation_steps,
                        'output_path': v.output_path,
                        'created_at': v.created_at.isoformat(),
                        'metadata': v.metadata
                    } for k, v in self.lineage_registry.items()
                },
                'quality_rules': {
                    k: {
                        'name': v.name,
                        'rule_type': v.rule_type,
                        'severity': v.severity,
                        'description': v.description
                    } for k, v in self.quality_rules.items()
                },
                'quality_reports': {
                    k: {
                        'dataset_id': v.dataset_id,
                        'timestamp': v.timestamp.isoformat(),
                        'total_records': v.total_records,
                        'passed_records': v.passed_records,
                        'failed_records': v.failed_records,
                        'quality_score': v.quality_score,
                        'rule_violations': v.rule_violations
                    } for k, v in self.quality_reports.items()
                },
                'metadata_registry': self.metadata_registry,
                'compliance_rules': self.compliance_rules,
                'compliance_reports': self.compliance_reports
            }
            
            # Write to file
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported governance data to: {export_path}")
            return True
            
        except Exception as e:
            await self.error_handler.handle_error(
                "governance_export_error",
                f"Failed to export governance data: {str(e)}",
                error_data={'export_path': export_path, 'error': str(e)}
            )
            return False
