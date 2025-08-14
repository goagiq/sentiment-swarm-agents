"""
Decision Support System Configuration

Provides configuration parameters for the enhanced decision support system
including knowledge graph integration, multilingual support, and real-time
data integration capabilities.
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum


class DecisionContextType(Enum):
    """Types of decision contexts that can be extracted."""
    BUSINESS_OBJECTIVES = "business_objectives"
    MARKET_CONDITIONS = "market_conditions"
    RESOURCE_CONSTRAINTS = "resource_constraints"
    STAKEHOLDER_PREFERENCES = "stakeholder_preferences"
    HISTORICAL_PATTERNS = "historical_patterns"
    REAL_TIME_DATA = "real_time_data"
    EXTERNAL_SYSTEM_DATA = "external_system_data"


class ConfidenceSource(Enum):
    """Sources of confidence scoring."""
    KNOWLEDGE_GRAPH = "knowledge_graph"
    HISTORICAL_PATTERNS = "historical_patterns"
    REAL_TIME_DATA = "real_time_data"
    MULTI_MODAL_ANALYSIS = "multi_modal_analysis"
    EXTERNAL_SYSTEMS = "external_systems"
    USER_FEEDBACK = "user_feedback"


class DecisionPatternType(Enum):
    """Types of decision patterns that can be analyzed."""
    SUCCESS_PATTERNS = "success_patterns"
    FAILURE_PATTERNS = "failure_patterns"
    RISK_PATTERNS = "risk_patterns"
    OPPORTUNITY_PATTERNS = "opportunity_patterns"
    TEMPORAL_PATTERNS = "temporal_patterns"
    CULTURAL_PATTERNS = "cultural_patterns"


@dataclass
class KnowledgeGraphConfig:
    """Configuration for knowledge graph integration."""
    # Entity extraction settings
    entity_types: List[str] = field(default_factory=lambda: [
        "PERSON", "ORGANIZATION", "PRODUCT", "MARKET", "TECHNOLOGY",
        "PROCESS", "RISK", "OPPORTUNITY", "CONSTRAINT", "GOAL"
    ])
    
    # Relationship mapping settings
    relationship_types: List[str] = field(default_factory=lambda: [
        "INFLUENCES", "DEPENDS_ON", "COMPETES_WITH", "SUPPORTS",
        "CONFLICTS_WITH", "ENABLES", "CONSTRAINS", "ALIGNS_WITH"
    ])
    
    # Query settings
    max_entities_per_query: int = 50
    max_relationships_per_entity: int = 20
    similarity_threshold: float = 0.7
    
    # Context extraction settings
    context_depth: int = 3  # How many levels deep to explore relationships
    context_time_window: str = "1_year"  # Time window for historical context
    
    # Pattern analysis settings
    pattern_min_occurrences: int = 3
    pattern_confidence_threshold: float = 0.6


@dataclass
class MultilingualDecisionConfig:
    """Configuration for multilingual decision support."""
    # Language-specific decision patterns
    language_patterns: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "en": {
            "decision_style": "analytical",
            "risk_tolerance": "moderate",
            "time_orientation": "future",
            "cultural_factors": ["individualism", "low_context"]
        },
        "zh": {
            "decision_style": "holistic",
            "risk_tolerance": "conservative",
            "time_orientation": "long_term",
            "cultural_factors": [
                "collectivism", "high_context", "guanxi"
            ]
        },
        "ja": {
            "decision_style": "consensus_based",
            "risk_tolerance": "conservative",
            "time_orientation": "long_term",
            "cultural_factors": ["collectivism", "high_context", "wa"]
        },
        "ko": {
            "decision_style": "hierarchical",
            "risk_tolerance": "moderate",
            "time_orientation": "medium_term",
            "cultural_factors": ["collectivism", "high_context", "jeong"]
        },
        "ar": {
            "decision_style": "relationship_based",
            "risk_tolerance": "moderate",
            "time_orientation": "present",
            "cultural_factors": ["collectivism", "high_context", "wasta"]
        },
        "ru": {
            "decision_style": "authoritative",
            "risk_tolerance": "high",
            "time_orientation": "present",
            "cultural_factors": ["collectivism", "high_context", "blat"]
        },
        "es": {
            "decision_style": "relationship_based",
            "risk_tolerance": "moderate",
            "time_orientation": "present",
            "cultural_factors": ["collectivism", "medium_context", "personalismo"]
        },
        "fr": {
            "decision_style": "analytical",
            "risk_tolerance": "moderate",
            "time_orientation": "future",
            "cultural_factors": ["individualism", "medium_context", "rationality"]
        },
        "de": {
            "decision_style": "systematic",
            "risk_tolerance": "low",
            "time_orientation": "long_term",
            "cultural_factors": ["individualism", "low_context", "precision"]
        },
        "it": {
            "decision_style": "flexible",
            "risk_tolerance": "moderate",
            "time_orientation": "present",
            "cultural_factors": ["collectivism", "medium_context", "personal_relationships"]
        }
    })
    
    # Language-specific entity extraction patterns
    entity_patterns: Dict[str, Dict[str, List[str]]] = field(default_factory=lambda: {
        "en": {
            "business_terms": ["strategy", "objective", "goal", "target", "metric"],
            "risk_terms": ["risk", "threat", "vulnerability", "exposure"],
            "opportunity_terms": ["opportunity", "potential", "prospect", "advantage"]
        },
        "zh": {
            "business_terms": ["战略", "目标", "目的", "指标", "计划"],
            "risk_terms": ["风险", "威胁", "漏洞", "暴露"],
            "opportunity_terms": ["机会", "潜力", "前景", "优势"]
        },
        "ja": {
            "business_terms": ["戦略", "目標", "目的", "指標", "計画"],
            "risk_terms": ["リスク", "脅威", "脆弱性", "露出"],
            "opportunity_terms": ["機会", "可能性", "見込み", "利点"]
        },
        "ko": {
            "business_terms": ["전략", "목표", "목적", "지표", "계획"],
            "risk_terms": ["위험", "위협", "취약점", "노출"],
            "opportunity_terms": ["기회", "잠재력", "전망", "이점"]
        },
        "ar": {
            "business_terms": ["استراتيجية", "هدف", "غاية", "مؤشر", "خطة"],
            "risk_terms": ["خطر", "تهديد", "ضعف", "تعرض"],
            "opportunity_terms": ["فرصة", "إمكانية", "آفاق", "ميزة"]
        },
        "ru": {
            "business_terms": ["стратегия", "цель", "задача", "показатель", "план"],
            "risk_terms": ["риск", "угроза", "уязвимость", "воздействие"],
            "opportunity_terms": ["возможность", "потенциал", "перспектива", "преимущество"]
        }
    })
    
    # Language-specific decision reasoning patterns
    reasoning_patterns: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "en": {
            "cause_effect": "because {cause}, {effect}",
            "evidence": "based on {evidence}, {conclusion}",
            "comparison": "compared to {baseline}, {current} shows {difference}",
            "prediction": "given {conditions}, {outcome} is likely"
        },
        "zh": {
            "cause_effect": "由于{cause}，{effect}",
            "evidence": "基于{evidence}，{conclusion}",
            "comparison": "与{baseline}相比，{current}显示{difference}",
            "prediction": "考虑到{conditions}，{outcome}很可能发生"
        },
        "ja": {
            "cause_effect": "{cause}により、{effect}",
            "evidence": "{evidence}に基づいて、{conclusion}",
            "comparison": "{baseline}と比較して、{current}は{difference}を示している",
            "prediction": "{conditions}を考慮すると、{outcome}が起こる可能性が高い"
        }
    })


@dataclass
class RealTimeDataConfig:
    """Configuration for real-time data integration."""
    # Data source settings
    enabled_sources: List[str] = field(default_factory=lambda: [
        "market_data", "social_media", "news_feeds", "iot_sensors"
    ])
    
    # Update frequency settings
    update_intervals: Dict[str, int] = field(default_factory=lambda: {
        "market_data": 30,  # seconds
        "social_media": 60,  # seconds
        "news_feeds": 300,   # seconds
        "iot_sensors": 10    # seconds
    })
    
    # Data retention settings
    retention_periods: Dict[str, str] = field(default_factory=lambda: {
        "market_data": "7_days",
        "social_media": "30_days",
        "news_feeds": "90_days",
        "iot_sensors": "1_year"
    })
    
    # Processing settings
    batch_size: int = 100
    max_concurrent_streams: int = 10
    buffer_size: int = 1000


@dataclass
class ExplainableAIConfig:
    """Configuration for explainable AI capabilities."""
    # Explanation generation settings
    explanation_depth: str = "detailed"  # simple, detailed, comprehensive
    include_evidence: bool = True
    include_confidence_factors: bool = True
    include_alternatives: bool = True
    
    # Evidence collection settings
    evidence_sources: List[str] = field(default_factory=lambda: [
        "knowledge_graph", "historical_data", "real_time_data",
        "multi_modal_analysis", "external_systems"
    ])
    
    # Reasoning engine settings
    reasoning_models: List[str] = field(default_factory=lambda: [
        "causal_reasoning", "temporal_reasoning", "comparative_reasoning"
    ])
    
    # Output format settings
    output_formats: List[str] = field(default_factory=lambda: [
        "natural_language", "structured", "visual"
    ])


@dataclass
class ConfidenceScoringConfig:
    """Configuration for confidence scoring."""
    # Scoring weights
    source_weights: Dict[str, float] = field(default_factory=lambda: {
        "knowledge_graph": 0.25,
        "historical_patterns": 0.20,
        "real_time_data": 0.15,
        "multi_modal_analysis": 0.20,
        "external_systems": 0.10,
        "user_feedback": 0.10
    })
    
    # Reliability assessment settings
    reliability_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "high": 0.8,
        "medium": 0.6,
        "low": 0.4
    })
    
    # Temporal confidence settings
    temporal_decay_rate: float = 0.1  # per day
    max_temporal_age: int = 30  # days
    
    # Cross-validation settings
    cross_validation_sources: int = 3
    minimum_agreement_threshold: float = 0.7


@dataclass
class DecisionSupportConfig:
    """Main configuration for the decision support system."""
    
    # Core settings
    knowledge_graph: KnowledgeGraphConfig = field(default_factory=KnowledgeGraphConfig)
    multilingual: MultilingualDecisionConfig = field(default_factory=MultilingualDecisionConfig)
    real_time_data: RealTimeDataConfig = field(default_factory=RealTimeDataConfig)
    explainable_ai: ExplainableAIConfig = field(default_factory=ExplainableAIConfig)
    confidence_scoring: ConfidenceScoringConfig = field(default_factory=ConfidenceScoringConfig)
    
    # General settings
    max_recommendations: int = 10
    recommendation_confidence_threshold: float = 0.6
    context_extraction_timeout: int = 30  # seconds
    decision_history_retention: str = "2_years"
    
    # Performance settings
    cache_ttl: int = 3600  # seconds
    max_concurrent_requests: int = 20
    request_timeout: int = 60  # seconds
    
    # Monitoring settings
    enable_decision_monitoring: bool = True
    monitoring_interval: int = 300  # seconds
    alert_threshold: float = 0.3  # deviation threshold
    
    # Integration settings
    enable_external_integration: bool = True
    external_system_timeout: int = 30  # seconds
    max_external_requests: int = 10


# Global configuration instance
decision_support_config = DecisionSupportConfig()


def get_decision_support_config() -> DecisionSupportConfig:
    """Get the global decision support configuration."""
    return decision_support_config


def get_language_decision_config(language: str) -> Dict[str, Any]:
    """Get language-specific decision configuration."""
    config = decision_support_config.multilingual
    return config.language_patterns.get(language, config.language_patterns["en"])


def get_language_entity_patterns(language: str) -> Dict[str, List[str]]:
    """Get language-specific entity extraction patterns."""
    config = decision_support_config.multilingual
    return config.entity_patterns.get(language, config.entity_patterns["en"])


def get_language_reasoning_patterns(language: str) -> Dict[str, str]:
    """Get language-specific reasoning patterns."""
    config = decision_support_config.multilingual
    return config.reasoning_patterns.get(language, config.reasoning_patterns["en"])


def get_knowledge_graph_config() -> KnowledgeGraphConfig:
    """Get knowledge graph configuration."""
    return decision_support_config.knowledge_graph


def get_real_time_config() -> RealTimeDataConfig:
    """Get real-time data configuration."""
    return decision_support_config.real_time_data


def get_explainable_ai_config() -> ExplainableAIConfig:
    """Get explainable AI configuration."""
    return decision_support_config.explainable_ai


def get_confidence_scoring_config() -> ConfidenceScoringConfig:
    """Get confidence scoring configuration."""
    return decision_support_config.confidence_scoring
