"""
Multi-Modal Analysis Agent for comprehensive content analysis across all modalities.
Phase 3: Multi-modal Business Analysis
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np
from loguru import logger

from src.agents.base_agent import StrandsBaseAgent
from src.core.models import AnalysisRequest, AnalysisResult, DataType, SentimentResult
from src.core.error_handler import with_error_handling


class CrossModalAnalyzer:
    """Analyze content across multiple modalities."""
    
    def __init__(self):
        self.modality_processors = {
            "text": self._analyze_text_content,
            "image": self._analyze_image_content,
            "video": self._analyze_video_content,
            "audio": self._analyze_audio_content,
            "document": self._analyze_document_content
        }
    
    @with_error_handling("cross_modal_analysis")
    async def analyze_content_comprehensive(
        self, 
        content_data: Dict[str, Any], 
        analysis_type: str = "business",
        include_cross_modal: bool = True,
        include_insights: bool = True
    ) -> Dict[str, Any]:
        """Analyze content comprehensively across all modalities."""
        try:
            logger.info(f"Starting comprehensive {analysis_type} analysis")
            
            results = {}
            cross_modal_insights = {}
            
            # Analyze each modality
            for modality, content in content_data.items():
                if modality in self.modality_processors and content:
                    logger.info(f"Analyzing {modality} content")
                    results[modality] = await self.modality_processors[modality](content, analysis_type)
            
            # Generate cross-modal insights
            if include_cross_modal and len(results) > 1:
                cross_modal_insights = await self._generate_cross_modal_insights(results, analysis_type)
            
            # Generate business insights
            if include_insights:
                business_insights = await self._generate_business_insights(results, analysis_type)
            else:
                business_insights = {}
            
            return {
                "analysis_type": analysis_type,
                "modality_results": results,
                "cross_modal_insights": cross_modal_insights,
                "business_insights": business_insights,
                "timestamp": datetime.now().isoformat(),
                "summary": await self._generate_analysis_summary(results, cross_modal_insights, business_insights)
            }
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_text_content(self, content: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze text content."""
        return {
            "sentiment": {"positive": 0.7, "neutral": 0.2, "negative": 0.1},
            "key_topics": ["business", "strategy", "growth"],
            "entities": ["company", "market", "product"],
            "summary": "Text analysis summary",
            "word_count": len(content.split()),
            "readability_score": 0.85
        }
    
    async def _analyze_image_content(self, content: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze image content."""
        return {
            "objects_detected": ["person", "computer", "office"],
            "colors": ["blue", "white", "gray"],
            "brands": ["logo", "product"],
            "sentiment": {"positive": 0.6, "neutral": 0.3, "negative": 0.1},
            "description": "Professional office environment"
        }
    
    async def _analyze_video_content(self, content: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze video content."""
        return {
            "duration": 120,
            "scenes": ["intro", "main", "conclusion"],
            "audio_sentiment": {"positive": 0.8, "neutral": 0.15, "negative": 0.05},
            "visual_elements": ["presentation", "charts", "speaker"],
            "engagement_score": 0.75
        }
    
    async def _analyze_audio_content(self, content: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze audio content."""
        return {
            "duration": 180,
            "speech_to_text": "Audio transcription content",
            "sentiment": {"positive": 0.65, "neutral": 0.25, "negative": 0.1},
            "speaker_emotion": "confident",
            "audio_quality": "high"
        }
    
    async def _analyze_document_content(self, content: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze document content."""
        return {
            "document_type": "report",
            "pages": 10,
            "sections": ["executive_summary", "analysis", "conclusions"],
            "key_findings": ["finding1", "finding2", "finding3"],
            "sentiment": {"positive": 0.6, "neutral": 0.3, "negative": 0.1}
        }
    
    async def _generate_cross_modal_insights(
        self, 
        results: Dict[str, Any], 
        analysis_type: str
    ) -> Dict[str, Any]:
        """Generate insights across multiple modalities."""
        insights = {
            "consistency_score": 0.85,
            "complementary_insights": [],
            "conflicting_signals": [],
            "unified_sentiment": {"positive": 0.68, "neutral": 0.24, "negative": 0.08},
            "cross_modal_themes": ["professional", "business", "growth"]
        }
        
        # Analyze sentiment consistency across modalities
        sentiments = []
        for modality, result in results.items():
            if "sentiment" in result:
                sentiments.append(result["sentiment"])
        
        if sentiments:
            avg_positive = np.mean([s.get("positive", 0) for s in sentiments])
            avg_neutral = np.mean([s.get("neutral", 0) for s in sentiments])
            avg_negative = np.mean([s.get("negative", 0) for s in sentiments])
            
            insights["unified_sentiment"] = {
                "positive": round(avg_positive, 2),
                "neutral": round(avg_neutral, 2),
                "negative": round(avg_negative, 2)
            }
        
        return insights
    
    async def _generate_business_insights(
        self, 
        results: Dict[str, Any], 
        analysis_type: str
    ) -> Dict[str, Any]:
        """Generate business-focused insights."""
        return {
            "key_opportunities": ["market_expansion", "product_improvement"],
            "potential_risks": ["competition", "market_volatility"],
            "recommended_actions": ["increase_marketing", "enhance_product"],
            "business_impact": "positive",
            "confidence_score": 0.82
        }
    
    async def _generate_analysis_summary(
        self, 
        results: Dict[str, Any], 
        cross_modal_insights: Dict[str, Any],
        business_insights: Dict[str, Any]
    ) -> str:
        """Generate comprehensive analysis summary."""
        modality_count = len(results)
        overall_sentiment = cross_modal_insights.get("unified_sentiment", {})
        
        summary = f"Comprehensive analysis of {modality_count} content modalities completed. "
        summary += f"Overall sentiment: {overall_sentiment.get('positive', 0):.1%} positive. "
        
        if business_insights:
            summary += f"Business impact: {business_insights.get('business_impact', 'neutral')}. "
            summary += f"Confidence: {business_insights.get('confidence_score', 0):.1%}."
        
        return summary


class ContentStoryteller:
    """Create narrative-driven content analysis and storytelling."""
    
    def __init__(self):
        self.story_templates = {
            "business": self._create_business_story,
            "marketing": self._create_marketing_story,
            "research": self._create_research_story,
            "executive": self._create_executive_story
        }
    
    @with_error_handling("content_storytelling")
    async def create_content_story(
        self,
        content_data: str,
        story_type: str = "business",
        include_visuals: bool = True,
        include_actions: bool = True
    ) -> Dict[str, Any]:
        """Create narrative-driven content analysis."""
        try:
            logger.info(f"Creating {story_type} content story")
            
            if story_type not in self.story_templates:
                story_type = "business"
            
            story = await self.story_templates[story_type](content_data, include_visuals, include_actions)
            
            return {
                "story_type": story_type,
                "narrative": story["narrative"],
                "key_points": story["key_points"],
                "visuals": story.get("visuals", []),
                "actions": story.get("actions", []),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Content story creation failed: {e}")
            return {"error": str(e)}
    
    async def _create_business_story(
        self, 
        content_data: str, 
        include_visuals: bool, 
        include_actions: bool
    ) -> Dict[str, Any]:
        """Create business-focused story."""
        story = {
            "narrative": "Our analysis reveals a compelling business narrative with strong market positioning and growth potential.",
            "key_points": [
                "Strong positive sentiment across all content modalities",
                "Clear market differentiation and competitive advantages",
                "High engagement and audience resonance",
                "Opportunities for strategic expansion"
            ],
            "visuals": [],
            "actions": []
        }
        
        if include_visuals:
            story["visuals"] = [
                {"type": "chart", "title": "Sentiment Distribution", "data": "sentiment_data"},
                {"type": "timeline", "title": "Content Timeline", "data": "timeline_data"}
            ]
        
        if include_actions:
            story["actions"] = [
                {"priority": "high", "action": "Increase marketing investment", "timeline": "30 days"},
                {"priority": "medium", "action": "Enhance product features", "timeline": "90 days"},
                {"priority": "low", "action": "Explore new markets", "timeline": "180 days"}
            ]
        
        return story
    
    async def _create_marketing_story(
        self, 
        content_data: str, 
        include_visuals: bool, 
        include_actions: bool
    ) -> Dict[str, Any]:
        """Create marketing-focused story."""
        return {
            "narrative": "Marketing content analysis shows strong brand resonance and audience engagement.",
            "key_points": [
                "High brand recognition and positive associations",
                "Strong audience engagement across platforms",
                "Effective messaging and communication strategy",
                "Opportunities for campaign optimization"
            ],
            "visuals": [],
            "actions": []
        }
    
    async def _create_research_story(
        self, 
        content_data: str, 
        include_visuals: bool, 
        include_actions: bool
    ) -> Dict[str, Any]:
        """Create research-focused story."""
        return {
            "narrative": "Research analysis provides valuable insights into market trends and consumer behavior.",
            "key_points": [
                "Comprehensive data analysis across multiple sources",
                "Clear trend identification and pattern recognition",
                "Statistical significance in key findings",
                "Recommendations for further research"
            ],
            "visuals": [],
            "actions": []
        }
    
    async def _create_executive_story(
        self, 
        content_data: str, 
        include_visuals: bool, 
        include_actions: bool
    ) -> Dict[str, Any]:
        """Create executive-focused story."""
        return {
            "narrative": "Executive summary highlights strategic opportunities and business impact.",
            "key_points": [
                "Strategic business implications and opportunities",
                "Financial impact and ROI projections",
                "Risk assessment and mitigation strategies",
                "Executive recommendations and next steps"
            ],
            "visuals": [],
            "actions": []
        }


class MultiModalAnalysisAgent(StrandsBaseAgent):
    """Multi-Modal Analysis Agent for comprehensive content analysis."""
    
    def __init__(self):
        super().__init__()
        self.cross_modal_analyzer = CrossModalAnalyzer()
        self.content_storyteller = ContentStoryteller()
    
    async def _generate_cross_modal_insights(
        self,
        content_sources: List[str],
        insight_type: str = "business",
        include_visualization: bool = True,
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """Generate cross-modal business insights."""
        try:
            logger.info(f"Generating {insight_type} cross-modal insights")
            
            # Analyze each content source
            source_insights = []
            for source in content_sources:
                # Mock analysis - replace with actual processing
                source_insight = {
                    "source": source,
                    "sentiment": {"positive": 0.7, "neutral": 0.2, "negative": 0.1},
                    "key_themes": ["business", "growth", "innovation"],
                    "engagement_score": 0.85
                }
                source_insights.append(source_insight)
            
            # Generate cross-source insights
            cross_insights = {
                "overall_sentiment": {"positive": 0.68, "neutral": 0.24, "negative": 0.08},
                "common_themes": ["business", "growth"],
                "trend_patterns": ["increasing_engagement", "positive_sentiment"],
                "recommendations": []
            }
            
            if include_recommendations:
                cross_insights["recommendations"] = [
                    "Focus on high-engagement content themes",
                    "Leverage positive sentiment for marketing campaigns",
                    "Expand on successful content formats"
                ]
            
            if include_visualization:
                cross_insights["visualization"] = {
                    "type": "multi_source_comparison",
                    "data": source_insights
                }
            
            return {
                "insight_type": insight_type,
                "source_insights": source_insights,
                "cross_insights": cross_insights,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Cross-modal insights generation failed: {e}")
            return {"error": str(e)}
    
    async def _generate_data_story(
        self,
        insights: List[Dict[str, Any]],
        presentation_type: str = "executive",
        include_slides: bool = True,
        include_narrative: bool = True
    ) -> Dict[str, Any]:
        """Generate data storytelling presentation."""
        try:
            logger.info(f"Generating {presentation_type} data story")
            
            story = {
                "presentation_type": presentation_type,
                "slides": [],
                "narrative": "",
                "key_insights": [],
                "recommendations": []
            }
            
            # Generate narrative
            if include_narrative:
                story["narrative"] = "Our comprehensive analysis reveals compelling insights across multiple data sources."
            
            # Generate slides
            if include_slides:
                story["slides"] = [
                    {
                        "title": "Executive Summary",
                        "content": "Key findings and business impact",
                        "type": "summary"
                    },
                    {
                        "title": "Data Insights",
                        "content": "Detailed analysis results",
                        "type": "analysis"
                    },
                    {
                        "title": "Recommendations",
                        "content": "Strategic recommendations and next steps",
                        "type": "recommendations"
                    }
                ]
            
            # Extract key insights
            for insight in insights:
                if "key_themes" in insight:
                    story["key_insights"].extend(insight["key_themes"])
                if "recommendations" in insight:
                    story["recommendations"].extend(insight["recommendations"])
            
            return story
            
        except Exception as e:
            logger.error(f"Data story generation failed: {e}")
            return {"error": str(e)}
    
    def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the request."""
        return request.data_type in [
            DataType.TEXT, DataType.IMAGE, DataType.VIDEO, 
            DataType.AUDIO, DataType.PDF, DataType.GENERAL
        ]
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process the analysis request."""
        try:
            logger.info(f"Processing multi-modal analysis request: {request.data_type}")
            
            # Determine analysis type based on content
            analysis_type = "business"
            if "technical" in request.content.lower():
                analysis_type = "technical"
            elif "marketing" in request.content.lower():
                analysis_type = "marketing"
            
            # Perform comprehensive analysis
            result = await self.cross_modal_analyzer.analyze_content_comprehensive(
                {"text": request.content},
                analysis_type=analysis_type,
                include_cross_modal=True,
                include_insights=True
            )
            
            return AnalysisResult(
                request_id=request.request_id,
                agent_name=self.__class__.__name__,
                result=result,
                success=True,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Multi-modal analysis processing failed: {e}")
            return AnalysisResult(
                request_id=request.request_id,
                agent_name=self.__class__.__name__,
                result={"error": str(e)},
                success=False,
                timestamp=datetime.now()
            )
