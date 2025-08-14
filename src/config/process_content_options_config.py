"""
Configuration for process_content tool options based on question types and scenarios.
This prevents the recurring 'Invalid type for parameter option' error by providing
properly formatted options for different types of content processing requests.
"""

from typing import Dict, Any, List, Optional
from enum import Enum
import re

try:
    from pydantic import BaseModel, Field
except ImportError:
    # Fallback for environments without pydantic
    class BaseModel:
        pass
    
    def Field(*args, **kwargs):
        return None


class QuestionCategory(str, Enum):
    """Categories of questions for automatic options configuration."""
    STRATEGIC_ANALYSIS = "strategic_analysis"
    CYBER_WARFARE = "cyber_warfare"
    INFORMATION_OPERATIONS = "information_operations"
    BUSINESS_INTELLIGENCE = "business_intelligence"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    ENTITY_EXTRACTION = "entity_extraction"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    DOCUMENT_ANALYSIS = "document_analysis"
    AUDIO_ANALYSIS = "audio_analysis"
    VIDEO_ANALYSIS = "video_analysis"
    GENERAL_QUERY = "general_query"


class ContentType(str, Enum):
    """Content types for processing."""
    TEXT = "text"
    PDF = "pdf"
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"
    WEBSITE = "website"
    AUTO = "auto"


class ProcessContentOptionsConfig(BaseModel):
    """Configuration for process_content tool options."""
    
    # Question type detection patterns
    question_patterns: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "strategic_analysis": [
                r"strategic.*principle",
                r"art of war",
                r"military.*strategy",
                r"tactical.*analysis",
                r"strategic.*thinking",
                r"warfare.*strategy",
                r"battle.*tactics",
                r"military.*doctrine"
            ],
            "cyber_warfare": [
                r"cyber.*warfare",
                r"cyber.*attack",
                r"cyber.*defense",
                r"cyber.*security",
                r"digital.*warfare",
                r"information.*warfare",
                r"cyber.*threat",
                r"cyber.*intelligence"
            ],
            "information_operations": [
                r"information.*operation",
                r"disinformation",
                r"propaganda",
                r"psychological.*operation",
                r"narrative.*control",
                r"cognitive.*warfare",
                r"social.*engineering",
                r"influence.*operation"
            ],
            "business_intelligence": [
                r"business.*intelligence",
                r"market.*analysis",
                r"competitive.*intelligence",
                r"business.*strategy",
                r"market.*research",
                r"industry.*analysis",
                r"business.*forecast",
                r"economic.*analysis"
            ],
            "sentiment_analysis": [
                r"sentiment.*analysis",
                r"emotion.*analysis",
                r"opinion.*mining",
                r"mood.*analysis",
                r"attitude.*analysis",
                r"feeling.*analysis",
                r"emotional.*tone",
                r"sentiment.*detection"
            ],
            "entity_extraction": [
                r"entity.*extraction",
                r"named.*entity",
                r"person.*extraction",
                r"organization.*extraction",
                r"location.*extraction",
                r"entity.*recognition",
                r"information.*extraction",
                r"data.*extraction"
            ],
            "knowledge_graph": [
                r"knowledge.*graph",
                r"semantic.*network",
                r"concept.*map",
                r"relationship.*mapping",
                r"entity.*relationship",
                r"knowledge.*base",
                r"semantic.*analysis",
                r"concept.*extraction"
            ],
            "document_analysis": [
                r"document.*analysis",
                r"text.*analysis",
                r"document.*processing",
                r"text.*mining",
                r"document.*extraction",
                r"content.*analysis",
                r"document.*classification",
                r"text.*classification"
            ],
            "audio_analysis": [
                r"audio.*analysis",
                r"speech.*analysis",
                r"audio.*processing",
                r"speech.*recognition",
                r"audio.*transcription",
                r"voice.*analysis",
                r"audio.*content",
                r"speech.*content"
            ],
            "video_analysis": [
                r"video.*analysis",
                r"video.*processing",
                r"video.*content",
                r"visual.*analysis",
                r"video.*extraction",
                r"video.*transcription",
                r"video.*intelligence",
                r"visual.*content"
            ]
        },
        description="Regex patterns to detect question categories"
    )
    
    # Default options for each category
    category_options: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "strategic_analysis": {
                "analysis_type": "strategic_intelligence",
                "focus_areas": [
                    "strategic_principles", "military_strategy", "tactical_analysis"
                ],
                "output_format": "comprehensive_analysis",
                "include_examples": True,
                "include_recommendations": True,
                "depth_level": "detailed"
            },
            "cyber_warfare": {
                "analysis_type": "cyber_intelligence",
                "focus_areas": ["cyber_warfare", "information_operations", "digital_strategy"],
                "output_format": "strategic_analysis",
                "include_threat_analysis": True,
                "include_defense_strategies": True,
                "depth_level": "comprehensive"
            },
            "information_operations": {
                "analysis_type": "information_intelligence",
                "focus_areas": ["information_operations", "psychological_operations", "narrative_control"],
                "output_format": "operational_analysis",
                "include_tactics": True,
                "include_countermeasures": True,
                "depth_level": "detailed"
            },
            "business_intelligence": {
                "analysis_type": "business_intelligence",
                "focus_areas": ["market_analysis", "competitive_intelligence", "business_strategy"],
                "output_format": "business_report",
                "include_insights": True,
                "include_recommendations": True,
                "depth_level": "comprehensive"
            },
            "sentiment_analysis": {
                "analysis_type": "sentiment_analysis",
                "focus_areas": ["emotion_detection", "opinion_mining", "sentiment_classification"],
                "output_format": "sentiment_report",
                "include_confidence_scores": True,
                "include_emotion_breakdown": True,
                "depth_level": "detailed"
            },
            "entity_extraction": {
                "analysis_type": "entity_extraction",
                "focus_areas": ["named_entity_recognition", "entity_linking", "relationship_extraction"],
                "output_format": "entity_report",
                "include_confidence_scores": True,
                "include_relationships": True,
                "depth_level": "comprehensive"
            },
            "knowledge_graph": {
                "analysis_type": "knowledge_graph",
                "focus_areas": ["concept_extraction", "relationship_mapping", "semantic_analysis"],
                "output_format": "knowledge_graph",
                "include_visualization": True,
                "include_metadata": True,
                "depth_level": "detailed"
            },
            "document_analysis": {
                "analysis_type": "document_analysis",
                "focus_areas": ["text_extraction", "content_analysis", "document_classification"],
                "output_format": "document_report",
                "include_summary": True,
                "include_key_points": True,
                "depth_level": "comprehensive"
            },
            "audio_analysis": {
                "analysis_type": "audio_analysis",
                "focus_areas": ["speech_recognition", "audio_transcription", "audio_content_analysis"],
                "output_format": "audio_report",
                "include_transcript": True,
                "include_metadata": True,
                "depth_level": "detailed"
            },
            "video_analysis": {
                "analysis_type": "video_analysis",
                "focus_areas": ["visual_analysis", "video_transcription", "video_content_analysis"],
                "output_format": "video_report",
                "include_transcript": True,
                "include_visual_elements": True,
                "depth_level": "comprehensive"
            },
            "general_query": {
                "analysis_type": "general_analysis",
                "focus_areas": ["general_information", "comprehensive_analysis"],
                "output_format": "general_report",
                "include_summary": True,
                "include_details": True,
                "depth_level": "standard"
            }
        },
        description="Default options for each question category"
    )
    
    # Content type detection patterns
    content_type_patterns: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "pdf": [r"\.pdf$", r"pdf.*document", r"pdf.*file"],
            "audio": [r"\.mp3$", r"\.wav$", r"\.m4a$", r"audio.*file", r"speech.*recording"],
            "video": [r"\.mp4$", r"\.avi$", r"\.mov$", r"video.*file", r"video.*recording"],
            "image": [r"\.jpg$", r"\.jpeg$", r"\.png$", r"\.gif$", r"image.*file", r"picture"],
            "website": [r"http[s]?://", r"www\.", r"website", r"web.*page", r"url"]
        },
        description="Patterns to detect content types"
    )


class ProcessContentOptionsManager:
    """Manager for process_content tool options."""
    
    def __init__(self, config: Optional[ProcessContentOptionsConfig] = None):
        """Initialize the options manager."""
        self.config = config or ProcessContentOptionsConfig()
    
    def detect_question_category(self, content: str) -> str:
        """Detect the category of a question based on content."""
        content_lower = content.lower()
        
        for category, patterns in self.config.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower, re.IGNORECASE):
                    return category
        
        return "general_query"
    
    def detect_content_type(self, content: str) -> str:
        """Detect the content type based on content."""
        content_lower = content.lower()
        
        for content_type, patterns in self.config.content_type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower, re.IGNORECASE):
                    return content_type
        
        return "text"
    
    def get_options_for_content(self, content: str, content_type: str = "auto") -> Optional[Dict[str, Any]]:
        """Get appropriate options for content processing."""
        try:
            # Detect question category
            category = self.detect_question_category(content)
            
            # Detect content type if auto
            if content_type == "auto":
                content_type = self.detect_content_type(content)
            
            # Get base options for category
            options = self.config.category_options.get(category, {}).copy()
            
            # Add content type specific options
            if content_type in ["pdf", "audio", "video", "image"]:
                options.update({
                    "extraction_method": "enhanced",
                    "include_metadata": True,
                    "preserve_formatting": True
                })
            
            # Add language detection if not specified
            if "language" not in options:
                options["language"] = "auto"
            
            return options if options else None
            
        except Exception as e:
            print(f"Error getting options for content: {e}")
            return None
    
    def get_safe_options(self, content: str, content_type: str = "auto") -> Optional[Dict[str, Any]]:
        """Get safe options that won't cause parameter type errors."""
        try:
            options = self.get_options_for_content(content, content_type)
            
            # Ensure options is a valid dict or None
            if options and isinstance(options, dict):
                # Validate and clean options
                cleaned_options = {}
                for key, value in options.items():
                    if isinstance(key, str) and value is not None:
                        cleaned_options[key] = value
                
                return cleaned_options if cleaned_options else None
            
            return None
            
        except Exception as e:
            print(f"Error getting safe options: {e}")
            return None


# Global instance for easy access
options_manager = ProcessContentOptionsManager()


def get_process_content_options(content: str, content_type: str = "auto") -> Optional[Dict[str, Any]]:
    """Convenience function to get process_content options."""
    return options_manager.get_safe_options(content, content_type)


def detect_question_type(content: str) -> str:
    """Convenience function to detect question type."""
    return options_manager.detect_question_category(content)


# Example usage and testing
if __name__ == "__main__":
    # Test the configuration
    test_questions = [
        "How do the strategic principles in The Art of War apply to modern cyber warfare?",
        "What is the sentiment of this customer review?",
        "Extract entities from this document",
        "Create a knowledge graph from this text",
        "Analyze this audio recording",
        "What is the weather like today?"
    ]
    
    for question in test_questions:
        category = detect_question_type(question)
        options = get_process_content_options(question)
        print(f"Question: {question[:50]}...")
        print(f"Category: {category}")
        print(f"Options: {options}")
        print("-" * 50)
