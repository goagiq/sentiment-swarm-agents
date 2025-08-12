"""
English language configuration for standard processing.
Provides baseline configuration for English content.
"""

from typing import Dict, List
from .base_config import BaseLanguageConfig, EntityPatterns, ProcessingSettings


class EnglishConfig(BaseLanguageConfig):
    """English language configuration with standard processing."""
    
    def __init__(self):
        super().__init__()
        self.language_code = "en"
        self.language_name = "English"
        self.entity_patterns = self.get_entity_patterns()
        self.processing_settings = self.get_processing_settings()
        self.relationship_templates = self.get_relationship_templates()
        self.detection_patterns = self.get_detection_patterns()
    
    def get_entity_patterns(self) -> EntityPatterns:
        """Get English entity patterns."""
        return EntityPatterns(
            person=[
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last
                r'\b[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z]\.\b',  # First M. L.
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Middle Last
            ],
            organization=[
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+Inc\.|\s+Corp\.|\s+Ltd\.|\s+LLC|\s+Company|\s+Corporation)\b',
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+University|\s+College|\s+Institute|\s+School)\b',
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+Government|\s+Agency|\s+Department|\s+Ministry)\b',
            ],
            location=[
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+City|\s+Town|\s+Village|\s+County|\s+State|\s+Country)\b',
                r'\b(?:New York|Los Angeles|Chicago|Houston|Phoenix|Philadelphia|San Antonio|San Diego|Dallas|San Jose)\b',
            ],
            concept=[
                r'\b(?:artificial intelligence|machine learning|deep learning|neural networks)\b',
                r'\b(?:blockchain|cloud computing|big data|internet of things|IoT)\b',
                r'\b(?:quantum computing|cybersecurity|data science|robotics)\b',
            ]
        )
    
    def get_processing_settings(self) -> ProcessingSettings:
        """Get English processing settings."""
        return ProcessingSettings(
            min_entity_length=2,
            max_entity_length=50,
            confidence_threshold=0.6,
            use_enhanced_extraction=True,
            relationship_prompt_simplified=False,  # English uses standard prompts
            use_hierarchical_relationships=False,
            entity_clustering_enabled=False,
            fallback_strategies=None
        )
    
    def get_relationship_templates(self) -> Dict[str, str]:
        """Get English relationship templates."""
        return {
            "person_organization": "WORKS_FOR",
            "person_location": "LOCATED_IN",
            "organization_location": "LOCATED_IN",
            "concept_concept": "RELATED_TO",
            "person_person": "RELATED_TO"
        }
    
    def get_detection_patterns(self) -> List[str]:
        """Get English language detection patterns."""
        return [
            r'\b(?:the|and|or|but|in|on|at|to|for|of|with|by|from|about|into|through|during|before|after|above|below)\b',  # Common prepositions
            r'\b(?:is|are|was|were|be|been|being|have|has|had|do|does|did|will|would|could|should|may|might|can)\b',  # Common verbs
            r'\b(?:a|an|the|this|that|these|those|my|your|his|her|its|our|their)\b',  # Common articles/determiners
            r'\b(?:I|you|he|she|it|we|they|me|him|her|us|them)\b',  # Common pronouns
        ]
