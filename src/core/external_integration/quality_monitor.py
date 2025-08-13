"""
Data Quality Monitor

Provides comprehensive data quality monitoring capabilities including:
- Data validation and cleaning
- Quality metrics calculation
- Anomaly detection
- Data profiling and statistics
- Quality reporting and alerting
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import re
import statistics
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class QualityMetric(Enum):
    """Data quality metrics"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"


class ValidationRule(Enum):
    """Validation rule types"""
    REQUIRED = "required"
    TYPE_CHECK = "type_check"
    RANGE_CHECK = "range_check"
    PATTERN_MATCH = "pattern_match"
    UNIQUE_CHECK = "unique_check"
    CUSTOM_FUNCTION = "custom_function"


@dataclass
class QualityRule:
    """Data quality rule"""
    field: str
    rule_type: ValidationRule
    parameters: Dict[str, Any] = field(default_factory=dict)
    severity: str = "error"  # error, warning, info
    description: str = ""


@dataclass
class QualityReport:
    """Data quality report"""
    timestamp: datetime
    dataset_name: str
    total_records: int
    valid_records: int
    invalid_records: int
    quality_score: float
    metrics: Dict[str, float]
    issues: List[Dict[str, Any]]
    recommendations: List[str]


class DataQualityMonitor:
    """
    Monitors and validates data quality across multiple sources
    """
    
    def __init__(self):
        """Initialize the data quality monitor"""
        self.quality_rules = {}
        self.reports = []
        self.quality_history = []
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for the quality monitor"""
        logging.basicConfig(level=logging.INFO)
        
    def add_quality_rules(self, dataset_name: str, rules: List[QualityRule]):
        """
        Add quality rules for a dataset
        
        Args:
            dataset_name: Name of the dataset
            rules: List of quality rules
        """
        self.quality_rules[dataset_name] = rules
        logger.info(f"Added {len(rules)} quality rules for dataset: {dataset_name}")
        
    def remove_quality_rules(self, dataset_name: str):
        """
        Remove quality rules for a dataset
        
        Args:
            dataset_name: Name of the dataset
        """
        if dataset_name in self.quality_rules:
            del self.quality_rules[dataset_name]
            logger.info(f"Removed quality rules for dataset: {dataset_name}")
            
    def _validate_field(self, value: Any, rule: QualityRule) -> Dict[str, Any]:
        """
        Validate a single field against a rule
        
        Args:
            value: Field value to validate
            rule: Quality rule to apply
            
        Returns:
            Validation result
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            if rule.rule_type == ValidationRule.REQUIRED:
                if value is None or (isinstance(value, str) and not value.strip()):
                    result['valid'] = False
                    result['errors'].append("Field is required")
                    
            elif rule.rule_type == ValidationRule.TYPE_CHECK:
                expected_type = rule.parameters.get('type')
                if expected_type and not isinstance(value, expected_type):
                    result['valid'] = False
                    result['errors'].append(f"Expected type {expected_type}")
                    
            elif rule.rule_type == ValidationRule.RANGE_CHECK:
                min_val = rule.parameters.get('min')
                max_val = rule.parameters.get('max')
                
                if min_val is not None and value < min_val:
                    result['valid'] = False
                    result['errors'].append(f"Value below minimum {min_val}")
                    
                if max_val is not None and value > max_val:
                    result['valid'] = False
                    result['errors'].append(f"Value above maximum {max_val}")
                    
            elif rule.rule_type == ValidationRule.PATTERN_MATCH:
                pattern = rule.parameters.get('pattern')
                if pattern and isinstance(value, str):
                    if not re.match(pattern, value):
                        result['valid'] = False
                        result['errors'].append(f"Value doesn't match pattern {pattern}")
                        
            elif rule.rule_type == ValidationRule.UNIQUE_CHECK:
                # This would need to be checked across the entire dataset
                # For now, we'll just note that uniqueness should be checked
                result['warnings'].append("Uniqueness check requires full dataset")
                
            elif rule.rule_type == ValidationRule.CUSTOM_FUNCTION:
                func = rule.parameters.get('function')
                if func and callable(func):
                    try:
                        if not func(value):
                            result['valid'] = False
                            result['errors'].append("Custom validation failed")
                    except Exception as e:
                        result['valid'] = False
                        result['errors'].append(f"Custom validation error: {str(e)}")
                        
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
            
        return result
        
    def _calculate_completeness(self, data: List[Dict[str, Any]], 
                               required_fields: List[str]) -> float:
        """
        Calculate completeness metric
        
        Args:
            data: Dataset
            required_fields: List of required fields
            
        Returns:
            Completeness score (0-1)
        """
        if not data:
            return 0.0
            
        total_fields = len(data) * len(required_fields)
        present_fields = 0
        
        for record in data:
            for field in required_fields:
                if field in record and record[field] is not None:
                    if isinstance(record[field], str) and record[field].strip():
                        present_fields += 1
                    elif not isinstance(record[field], str):
                        present_fields += 1
                        
        return present_fields / total_fields if total_fields > 0 else 0.0
        
    def _calculate_accuracy(self, data: List[Dict[str, Any]], 
                           validation_results: List[Dict[str, Any]]) -> float:
        """
        Calculate accuracy metric
        
        Args:
            data: Dataset
            validation_results: Validation results
            
        Returns:
            Accuracy score (0-1)
        """
        if not data:
            return 0.0
            
        total_validations = sum(len(result.get('field_results', {})) 
                               for result in validation_results)
        valid_validations = sum(
            sum(1 for field_result in result.get('field_results', {}).values() 
                if field_result.get('valid', False))
            for result in validation_results
        )
        
        return valid_validations / total_validations if total_validations > 0 else 0.0
        
    def _calculate_consistency(self, data: List[Dict[str, Any]]) -> float:
        """
        Calculate consistency metric
        
        Args:
            data: Dataset
            
        Returns:
            Consistency score (0-1)
        """
        if not data or len(data) < 2:
            return 1.0
            
        # Check for consistent data types and formats
        consistency_issues = 0
        total_checks = 0
        
        # Get all unique fields
        all_fields = set()
        for record in data:
            all_fields.update(record.keys())
            
        for field in all_fields:
            field_values = [record.get(field) for record in data 
                           if field in record and record[field] is not None]
            
            if len(field_values) > 1:
                # Check data type consistency
                types = set(type(val) for val in field_values)
                if len(types) > 1:
                    consistency_issues += 1
                total_checks += 1
                
        return 1.0 - (consistency_issues / total_checks) if total_checks > 0 else 1.0
        
    def _calculate_uniqueness(self, data: List[Dict[str, Any]], 
                             unique_fields: List[str]) -> float:
        """
        Calculate uniqueness metric
        
        Args:
            data: Dataset
            unique_fields: Fields that should be unique
            
        Returns:
            Uniqueness score (0-1)
        """
        if not data or not unique_fields:
            return 1.0
            
        total_duplicates = 0
        total_records = len(data)
        
        for field in unique_fields:
            field_values = [record.get(field) for record in data 
                           if field in record]
            unique_values = set(field_values)
            duplicates = len(field_values) - len(unique_values)
            total_duplicates += duplicates
            
        return 1.0 - (total_duplicates / (total_records * len(unique_fields)))
        
    def _detect_anomalies(self, data: List[Dict[str, Any]], 
                         numeric_fields: List[str]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in numeric fields
        
        Args:
            data: Dataset
            numeric_fields: List of numeric fields to check
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        for field in numeric_fields:
            values = []
            for record in data:
                if field in record and isinstance(record[field], (int, float)):
                    values.append(record[field])
                    
            if len(values) > 10:  # Need sufficient data for anomaly detection
                # Calculate statistics
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0
                
                # Detect outliers (beyond 2 standard deviations)
                for i, value in enumerate(values):
                    if std_val > 0 and abs(value - mean_val) > 2 * std_val:
                        anomalies.append({
                            'field': field,
                            'value': value,
                            'record_index': i,
                            'type': 'outlier',
                            'severity': 'warning'
                        })
                        
        return anomalies
        
    def _generate_recommendations(self, report: QualityReport) -> List[str]:
        """
        Generate recommendations based on quality report
        
        Args:
            report: Quality report
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if report.quality_score < 0.8:
            recommendations.append("Overall data quality needs improvement")
            
        if report.metrics.get('completeness', 1.0) < 0.9:
            recommendations.append("Address missing data issues")
            
        if report.metrics.get('accuracy', 1.0) < 0.9:
            recommendations.append("Review and fix data validation errors")
            
        if report.metrics.get('consistency', 1.0) < 0.9:
            recommendations.append("Standardize data formats and types")
            
        if report.metrics.get('uniqueness', 1.0) < 0.9:
            recommendations.append("Remove or resolve duplicate records")
            
        if len(report.issues) > 0:
            recommendations.append(f"Review {len(report.issues)} quality issues")
            
        return recommendations
        
    async def validate_dataset(self, dataset_name: str, 
                              data: List[Dict[str, Any]]) -> QualityReport:
        """
        Validate a dataset and generate quality report
        
        Args:
            dataset_name: Name of the dataset
            data: Dataset to validate
            
        Returns:
            Quality report
        """
        start_time = time.time()
        
        # Get quality rules for this dataset
        rules = self.quality_rules.get(dataset_name, [])
        
        # Validate each record
        validation_results = []
        total_records = len(data)
        valid_records = 0
        
        for i, record in enumerate(data):
            record_validation = {
                'record_index': i,
                'valid': True,
                'field_results': {},
                'errors': [],
                'warnings': []
            }
            
            # Apply rules to each field
            for rule in rules:
                if rule.field in record:
                    field_result = self._validate_field(record[rule.field], rule)
                    record_validation['field_results'][rule.field] = field_result
                    
                    if not field_result['valid']:
                        record_validation['valid'] = False
                        record_validation['errors'].extend(field_result['errors'])
                        
                    record_validation['warnings'].extend(field_result['warnings'])
                elif rule.rule_type == ValidationRule.REQUIRED:
                    # Field is required but missing
                    record_validation['valid'] = False
                    record_validation['errors'].append(f"Required field '{rule.field}' is missing")
                    record_validation['field_results'][rule.field] = {
                        'valid': False,
                        'errors': [f"Required field '{rule.field}' is missing"],
                        'warnings': []
                    }
                    
            if record_validation['valid']:
                valid_records += 1
                
            validation_results.append(record_validation)
            
        # Calculate quality metrics
        required_fields = [rule.field for rule in rules 
                          if rule.rule_type == ValidationRule.REQUIRED]
        unique_fields = [rule.field for rule in rules 
                        if rule.rule_type == ValidationRule.UNIQUE_CHECK]
        numeric_fields = [rule.field for rule in rules 
                         if rule.rule_type == ValidationRule.RANGE_CHECK]
        
        metrics = {
            'completeness': self._calculate_completeness(data, required_fields),
            'accuracy': self._calculate_accuracy(data, validation_results),
            'consistency': self._calculate_consistency(data),
            'uniqueness': self._calculate_uniqueness(data, unique_fields)
        }
        
        # Calculate overall quality score
        quality_score = sum(metrics.values()) / len(metrics)
        
        # Detect anomalies
        anomalies = self._detect_anomalies(data, numeric_fields)
        
        # Collect all issues
        issues = []
        for result in validation_results:
            if not result['valid']:
                issues.append({
                    'record_index': result['record_index'],
                    'errors': result['errors'],
                    'warnings': result['warnings']
                })
                
        issues.extend(anomalies)
        
        # Generate report
        report = QualityReport(
            timestamp=datetime.now(),
            dataset_name=dataset_name,
            total_records=total_records,
            valid_records=valid_records,
            invalid_records=total_records - valid_records,
            quality_score=quality_score,
            metrics=metrics,
            issues=issues,
            recommendations=[]
        )
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)
        
        # Store report
        self.reports.append(report)
        
        # Update history
        self.quality_history.append({
            'timestamp': report.timestamp,
            'dataset_name': dataset_name,
            'quality_score': quality_score,
            'total_records': total_records,
            'valid_records': valid_records
        })
        
        processing_time = time.time() - start_time
        logger.info(f"Quality validation completed for {dataset_name} in {processing_time:.2f}s")
        
        return report
        
    def get_quality_history(self, dataset_name: Optional[str] = None, 
                           days: int = 30) -> List[Dict[str, Any]]:
        """
        Get quality history for a dataset
        
        Args:
            dataset_name: Dataset name (optional)
            days: Number of days to look back
            
        Returns:
            Quality history
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        history = self.quality_history.copy()
        
        if dataset_name:
            history = [h for h in history if h['dataset_name'] == dataset_name]
            
        history = [h for h in history if h['timestamp'] >= cutoff_date]
        
        return sorted(history, key=lambda x: x['timestamp'])
        
    def get_latest_report(self, dataset_name: str) -> Optional[QualityReport]:
        """
        Get the latest quality report for a dataset
        
        Args:
            dataset_name: Dataset name
            
        Returns:
            Latest quality report or None
        """
        dataset_reports = [r for r in self.reports 
                          if r.dataset_name == dataset_name]
        
        if dataset_reports:
            return max(dataset_reports, key=lambda x: x.timestamp)
        return None
        
    def get_quality_summary(self) -> Dict[str, Any]:
        """
        Get overall quality summary
        
        Returns:
            Quality summary
        """
        if not self.reports:
            return {'error': 'No quality reports available'}
            
        latest_reports = {}
        for report in self.reports:
            if (report.dataset_name not in latest_reports or 
                report.timestamp > latest_reports[report.dataset_name].timestamp):
                latest_reports[report.dataset_name] = report
                
        total_records = sum(r.total_records for r in latest_reports.values())
        total_valid = sum(r.valid_records for r in latest_reports.values())
        avg_quality = sum(r.quality_score for r in latest_reports.values()) / len(latest_reports)
        
        return {
            'total_datasets': len(latest_reports),
            'total_records': total_records,
            'total_valid_records': total_valid,
            'overall_quality_score': avg_quality,
            'datasets': {
                name: {
                    'quality_score': report.quality_score,
                    'total_records': report.total_records,
                    'valid_records': report.valid_records,
                    'last_updated': report.timestamp.isoformat()
                }
                for name, report in latest_reports.items()
            }
        }
