"""
Data Quality Manager

Automated data quality management with:
- Data validation rules
- Quality metrics tracking
- Automated data cleaning
- Quality reporting
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import re

from loguru import logger

from src.config.big_data_config import get_big_data_config
from src.core.error_handling_service import ErrorHandlingService


@dataclass
class QualityRule:
    """Represents a data quality rule."""
    name: str
    rule_type: str  # 'validation', 'completeness', 'consistency', 'accuracy'
    field_name: str
    rule_function: Callable
    severity: str = 'warning'  # 'error', 'warning', 'info'
    description: str = ""


@dataclass
class QualityMetrics:
    """Data quality metrics."""
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    completeness_score: float = 0.0
    accuracy_score: float = 0.0
    consistency_score: float = 0.0
    overall_score: float = 0.0
    rule_violations: Dict[str, int] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Data quality report."""
    dataset_id: str
    timestamp: datetime
    metrics: QualityMetrics
    violations: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class DataQualityManager:
    """
    Automated data quality management.
    """
    
    def __init__(self):
        """Initialize data quality manager."""
        self.config = get_big_data_config()
        self.error_handler = ErrorHandlingService()
        
        # Quality rules
        self.quality_rules: Dict[str, QualityRule] = {}
        
        # Quality reports
        self.quality_reports: Dict[str, QualityReport] = {}
        
        # Data cleaning functions
        self.cleaning_functions: Dict[str, Callable] = {}
        
        logger.info("DataQualityManager initialized")
    
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
    
    async def add_cleaning_function(self, name: str, 
                                  cleaning_function: Callable) -> bool:
        """Add a data cleaning function."""
        try:
            self.cleaning_functions[name] = cleaning_function
            logger.info(f"Added cleaning function: {name}")
            return True
            
        except Exception as e:
            await self.error_handler.handle_error(
                "cleaning_function_error",
                f"Failed to add cleaning function: {str(e)}",
                error_data={'function_name': name, 'error': str(e)}
            )
            return False
    
    async def validate_data(self, dataset_id: str, 
                          data: List[Dict[str, Any]]) -> QualityReport:
        """Validate data quality using registered rules."""
        try:
            total_records = len(data)
            valid_records = 0
            invalid_records = 0
            violations = []
            rule_violations = {}
            
            # Initialize metrics
            metrics = QualityMetrics(total_records=total_records)
            
            # Apply each quality rule
            for rule_name, rule in self.quality_rules.items():
                rule_violation_count = 0
                
                for record_idx, record in enumerate(data):
                    try:
                        # Apply rule function
                        if not rule.rule_function(record):
                            invalid_records += 1
                            rule_violation_count += 1
                            
                            violations.append({
                                'rule_name': rule_name,
                                'record_index': record_idx,
                                'field_name': rule.field_name,
                                'severity': rule.severity,
                                'description': rule.description
                            })
                        else:
                            valid_records += 1
                            
                    except Exception as e:
                        logger.warning(
                            f"Rule {rule_name} failed for record {record_idx}: {str(e)}"
                        )
                        invalid_records += 1
                        rule_violation_count += 1
                
                rule_violations[rule_name] = rule_violation_count
            
            # Calculate quality scores
            metrics.valid_records = valid_records
            metrics.invalid_records = invalid_records
            metrics.rule_violations = rule_violations
            
            # Calculate completeness score
            completeness_violations = sum(
                count for rule_name, count in rule_violations.items()
                if self.quality_rules[rule_name].rule_type == 'completeness'
            )
            metrics.completeness_score = (
                (total_records - completeness_violations) / total_records * 100
            ) if total_records > 0 else 0
            
            # Calculate accuracy score
            accuracy_violations = sum(
                count for rule_name, count in rule_violations.items()
                if self.quality_rules[rule_name].rule_type == 'accuracy'
            )
            metrics.accuracy_score = (
                (total_records - accuracy_violations) / total_records * 100
            ) if total_records > 0 else 0
            
            # Calculate consistency score
            consistency_violations = sum(
                count for rule_name, count in rule_violations.items()
                if self.quality_rules[rule_name].rule_type == 'consistency'
            )
            metrics.consistency_score = (
                (total_records - consistency_violations) / total_records * 100
            ) if total_records > 0 else 0
            
            # Calculate overall score
            metrics.overall_score = (
                metrics.completeness_score + 
                metrics.accuracy_score + 
                metrics.consistency_score
            ) / 3
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                metrics, violations
            )
            
            # Create quality report
            report = QualityReport(
                dataset_id=dataset_id,
                timestamp=datetime.now(),
                metrics=metrics,
                violations=violations,
                recommendations=recommendations
            )
            
            # Store report
            self.quality_reports[dataset_id] = report
            
            logger.info(
                f"Data quality validation completed for {dataset_id}: "
                f"{metrics.overall_score:.2f}% overall score"
            )
            
            return report
            
        except Exception as e:
            await self.error_handler.handle_error(
                "data_validation_error",
                f"Failed to validate data: {str(e)}",
                error_data={'dataset_id': dataset_id, 'error': str(e)}
            )
            
            # Return empty report on error
            return QualityReport(
                dataset_id=dataset_id,
                timestamp=datetime.now(),
                metrics=QualityMetrics()
            )
    
    async def clean_data(self, data: List[Dict[str, Any]], 
                        cleaning_rules: List[str]) -> List[Dict[str, Any]]:
        """Clean data using registered cleaning functions."""
        try:
            cleaned_data = data.copy()
            
            for rule_name in cleaning_rules:
                if rule_name in self.cleaning_functions:
                    cleaning_function = self.cleaning_functions[rule_name]
                    
                    for i, record in enumerate(cleaned_data):
                        try:
                            cleaned_data[i] = cleaning_function(record)
                        except Exception as e:
                            logger.warning(
                                f"Cleaning function {rule_name} failed for record {i}: {str(e)}"
                            )
                else:
                    logger.warning(f"Cleaning function {rule_name} not found")
            
            logger.info(f"Data cleaning completed with {len(cleaning_rules)} rules")
            return cleaned_data
            
        except Exception as e:
            await self.error_handler.handle_error(
                "data_cleaning_error",
                f"Failed to clean data: {str(e)}",
                error_data={'cleaning_rules': cleaning_rules, 'error': str(e)}
            )
            return data
    
    async def _generate_recommendations(self, metrics: QualityMetrics, 
                                      violations: List[Dict[str, Any]]) -> List[str]:
        """Generate data quality recommendations."""
        recommendations = []
        
        # Overall score recommendations
        if metrics.overall_score < 80:
            recommendations.append(
                "Overall data quality is below 80%. Consider implementing "
                "additional data validation rules."
            )
        
        # Completeness recommendations
        if metrics.completeness_score < 90:
            recommendations.append(
                "Data completeness is below 90%. Review missing data patterns "
                "and implement data collection improvements."
            )
        
        # Accuracy recommendations
        if metrics.accuracy_score < 90:
            recommendations.append(
                "Data accuracy is below 90%. Review validation rules and "
                "consider data source improvements."
            )
        
        # Consistency recommendations
        if metrics.consistency_score < 90:
            recommendations.append(
                "Data consistency is below 90%. Review data format standards "
                "and implement consistency checks."
            )
        
        # Rule-specific recommendations
        rule_violation_counts = {}
        for violation in violations:
            rule_name = violation['rule_name']
            rule_violation_counts[rule_name] = rule_violation_counts.get(rule_name, 0) + 1
        
        for rule_name, count in rule_violation_counts.items():
            if count > len(violations) * 0.1:  # More than 10% violations
                recommendations.append(
                    f"Rule '{rule_name}' has {count} violations. "
                    "Consider reviewing and updating this rule."
                )
        
        return recommendations
    
    async def get_quality_summary(self) -> Dict[str, Any]:
        """Get quality management summary."""
        try:
            total_reports = len(self.quality_reports)
            total_rules = len(self.quality_rules)
            total_cleaning_functions = len(self.cleaning_functions)
            
            if total_reports > 0:
                avg_overall_score = sum(
                    report.metrics.overall_score for report in self.quality_reports.values()
                ) / total_reports
            else:
                avg_overall_score = 0
            
            return {
                'total_reports': total_reports,
                'total_rules': total_rules,
                'total_cleaning_functions': total_cleaning_functions,
                'average_overall_score': avg_overall_score,
                'recent_reports': list(self.quality_reports.keys())[-5:]  # Last 5
            }
            
        except Exception as e:
            await self.error_handler.handle_error(
                "quality_summary_error",
                f"Failed to get quality summary: {str(e)}",
                error_data={'error': str(e)}
            )
            return {}
    
    async def export_quality_report(self, dataset_id: str, 
                                  export_path: str) -> bool:
        """Export quality report to file."""
        try:
            if dataset_id not in self.quality_reports:
                logger.error(f"Quality report for {dataset_id} not found")
                return False
            
            report = self.quality_reports[dataset_id]
            
            export_data = {
                'dataset_id': report.dataset_id,
                'timestamp': report.timestamp.isoformat(),
                'metrics': {
                    'total_records': report.metrics.total_records,
                    'valid_records': report.metrics.valid_records,
                    'invalid_records': report.metrics.invalid_records,
                    'completeness_score': report.metrics.completeness_score,
                    'accuracy_score': report.metrics.accuracy_score,
                    'consistency_score': report.metrics.consistency_score,
                    'overall_score': report.metrics.overall_score,
                    'rule_violations': report.metrics.rule_violations
                },
                'violations': report.violations,
                'recommendations': report.recommendations
            }
            
            # Write to file
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported quality report to: {export_path}")
            return True
            
        except Exception as e:
            await self.error_handler.handle_error(
                "quality_report_export_error",
                f"Failed to export quality report: {str(e)}",
                error_data={'dataset_id': dataset_id, 'export_path': export_path, 'error': str(e)}
            )
            return False
