"""
Scenario Analysis Module

This module provides advanced what-if scenario analysis capabilities including
interactive scenario building, impact analysis, risk assessment, and scenario comparison.

Components:
- ScenarioBuilder: Interactive scenario creation and management
- ImpactAnalyzer: Scenario outcome prediction and impact quantification
- RiskAssessor: Scenario risk evaluation and scoring
- ScenarioComparator: Multi-scenario analysis and comparison
"""

from .scenario_builder import ScenarioBuilder
from .impact_analyzer import ImpactAnalyzer
from .risk_assessor import RiskAssessor
from .scenario_comparator import ScenarioComparator

__all__ = [
    'ScenarioBuilder',
    'ImpactAnalyzer', 
    'RiskAssessor',
    'ScenarioComparator'
]
