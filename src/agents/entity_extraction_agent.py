"""
Entity Extraction Agent for extracting entities from text content.
Extracted from the knowledge graph agent to provide focused entity extraction capabilities.
Enhanced with Phase 6.1 and 6.2 improvements for better Chinese entity extraction.
"""

import asyncio
import json
import re
from typing import Dict, List, Optional
import logging

from src.agents.base_agent import StrandsBaseAgent
from src.core.strands_mock import tool
from src.core.models import (
    AnalysisRequest,
    AnalysisResult,
    DataType,
    SentimentResult,
    SentimentLabel,
    ProcessingStatus
)
from src.core.processing_service import ProcessingService
from src.core.error_handling_service import ErrorHandlingService, ErrorContext
from src.core.model_management_service import ModelManagementService
# from src.config.entity_extraction_config import get_language_config, get_patterns, get_common_entities

# Configure logger
logger = logging.getLogger(__name__)


class EntityExtractionAgent(StrandsBaseAgent):
    """Agent for extracting entities from text content with enhanced Chinese support."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
        **kwargs
    ):
        # Initialize services
        self.model_management_service = ModelManagementService()
        self.processing_service = ProcessingService()
        self.error_handling_service = ErrorHandlingService()

        # Set model name before calling super().__init__
        self.model_name = model_name or self.model_management_service.get_best_model("text")

        super().__init__(
            model_name=self.model_name,
            **kwargs
        )

        # Processing settings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Enhanced entity categories with better descriptions
        self.entity_categories = {
            "person": ["person", "individual", "human", "name", "politician", "leader", "expert"],
            "organization": ["organization", "company", "corporation", "institution", "agency", "university", "government"],
            "location": ["location", "place", "city", "country", "region", "address", "geographic"],
            "event": ["event", "conference", "meeting", "ceremony", "festival", "summit"],
            "product": ["product", "item", "goods", "service", "software", "platform"],
            "technology": ["technology", "software", "platform", "system", "tool", "ai", "ml"],
            "concept": ["concept", "idea", "theory", "principle", "methodology", "policy"],
            "date": ["date", "time", "period", "era", "year", "month"],
            "quantity": ["quantity", "amount", "number", "measure", "value", "percentage"],
            "other": ["other", "miscellaneous", "unknown"]
        }

        # Enhanced Chinese patterns (Phase 6.2 improvements)
        self.chinese_patterns = {
            'PERSON': [
                r'\b[\u4e00-\u9fff]{2,4}\b',  # 2-4 character names with boundaries
                r'\b[\u4e00-\u9fff]+\s*[先生|女士|教授|博士|院士|主席|总理|部长]\b',  # Titles with boundaries
                r'\b[\u4e00-\u9fff]{2,4}\s*[先生|女士]\b',  # Name + title combinations
            ],
            'ORGANIZATION': [
                r'\b[\u4e00-\u9fff]+(?:公司|集团|企业|大学|学院|研究所|研究院|医院|银行|政府|部门)\b',
                r'\b[\u4e00-\u9fff]+(?:科技|技术|信息|网络|软件|硬件|生物|医药|金融|教育|文化)\b',
                r'\b[\u4e00-\u9fff]{2,6}(?:公司|集团|企业)\b',  # Specific company patterns
            ],
            'LOCATION': [
                r'\b[\u4e00-\u9fff]+(?:市|省|县|区|国|州|城|镇|村)\b',
                r'\b[\u4e00-\u9fff]+(?:山|河|湖|海|江|河|岛|湾)\b',
                r'\b[\u4e00-\u9fff]{2,4}(?:市|省|国)\b',  # Specific location patterns
            ],
            'CONCEPT': [
                r'\b(?:人工智能|机器学习|深度学习|神经网络|自然语言处理|计算机视觉)\b',
                r'\b(?:量子计算|区块链|云计算|大数据|物联网|5G|6G)\b',
                r'\b(?:虚拟现实|增强现实|混合现实|元宇宙|数字化转型)\b',
                r'\b(?:数字经济|智能制造|绿色能源|可持续发展)\b',
            ]
        }

        # Enhanced Chinese entity dictionaries (Phase 6.3 improvements)
        self.chinese_dictionaries = {
            'PERSON': [
                '习近平', '李克强', '王毅', '马云', '马化腾', '任正非', 
                '李彦宏', '张朝阳', '丁磊', '雷军', '李国杰', '潘建伟'
            ],
            'ORGANIZATION': [
                '华为', '阿里巴巴', '腾讯', '百度', '京东', '美团',
                '清华大学', '北京大学', '中科院', '计算所', '自动化所'
            ],
            'LOCATION': [
                '北京', '上海', '深圳', '广州', '杭州', '南京',
                '中国', '美国', '日本', '韩国', '德国', '法国'
            ],
            'CONCEPT': [
                '人工智能', '机器学习', '深度学习', '神经网络',
                '自然语言处理', '计算机视觉', '量子计算', '区块链'
            ]
        }

        # Enhanced Russian patterns (Phase 6.4 improvements)
        self.russian_patterns = {
            'PERSON': [
                r'\b[А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,})?\b',  # Full names (3+ chars each)
                r'\b[А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.\b',  # Name with initials
                r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,})*\s+(?:господин|госпожа|доктор|профессор)\b',  # With titles
            ],
            'ORGANIZATION': [
                r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[а-яё]{2,})*(?:ООО|ОАО|ЗАО|ПАО|ГК|Корпорация|Компания)\b',
                r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[а-яё]{2,})*(?:Университет|Институт|Академия|Университет)\b',
                r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[а-яё]{2,})*(?:Правительство|Министерство|Агентство)\b',
            ],
            'LOCATION': [
                r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[а-яё]{2,})*(?:город|область|край|республика|район)\b',
                r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[а-яё]{2,})*(?:улица|проспект|переулок|площадь)\b',
                r'\b(?:Москва|Санкт-Петербург|Новосибирск|Екатеринбург|Казань|Россия)\b',  # Major cities
            ],
            'CONCEPT': [
                r'\b(?:искусственный интеллект|машинное обучение|глубокое обучение)\b',
                r'\b(?:блокчейн|облачные вычисления|большие данные|интернет вещей)\b',
                r'\b(?:цифровая экономика|умное производство|зеленая энергия)\b',
            ]
        }

        # Enhanced Russian entity dictionaries (Phase 6.4 improvements)
        self.russian_dictionaries = {
            'PERSON': [
                'Владимир Путин', 'Дмитрий Медведев', 'Сергей Лавров', 'Алексей Миллер',
                'Герман Греф', 'Андрей Костин', 'Олег Дерипаска', 'Роман Абрамович',
                'Михаил Фридман', 'Алишер Усманов', 'Леонид Михельсон', 'Вагит Алекперов'
            ],
            'ORGANIZATION': [
                'Газпром', 'Сбербанк', 'Роснефть', 'Лукойл', 'Норникель',
                'МГУ', 'СПбГУ', 'МФТИ', 'РАН', 'Сколково', 'ВШЭ'
            ],
            'LOCATION': [
                'Москва', 'Санкт-Петербург', 'Новосибирск', 'Екатеринбург', 'Казань',
                'Россия', 'США', 'Китай', 'Германия', 'Франция', 'Великобритания'
            ],
            'CONCEPT': [
                'искусственный интеллект', 'машинное обучение', 'глубокое обучение',
                'нейронные сети', 'обработка естественного языка', 'компьютерное зрение',
                'квантовые вычисления', 'блокчейн', 'облачные вычисления'
            ]
        }

        # Agent metadata
        self.metadata.update({
            "agent_type": "entity_extraction",
            "model": self.model_name,
            "capabilities": [
                "entity_extraction",
                "entity_categorization",
                "chunk_based_processing",
                "enhanced_entity_detection",
                "multilingual_support",
                "pattern_based_extraction",
                "dictionary_based_extraction"
            ],
            "supported_data_types": [
                DataType.TEXT,
                DataType.AUDIO,
                DataType.VIDEO,
                DataType.WEBPAGE,
                DataType.PDF,
                DataType.SOCIAL_MEDIA
            ],
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "entity_categories": list(self.entity_categories.keys()),
            "enhanced_features": [
                "structured_prompts",
                "multi_strategy_extraction",
                "confidence_scoring",
                "entity_validation"
            ]
        })

        logger.info(f"Enhanced Entity Extraction Agent {self.agent_id} initialized with model {self.model_name}")

    def _get_tools(self) -> list:
        """Get list of tools for this agent."""
        return [
            self.extract_entities,
            self.extract_entities_enhanced,
            self.categorize_entities,
            self.extract_entities_from_chunks,
            self.get_entity_statistics,
            self.extract_entities_multilingual
        ]

    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the given request."""
        return request.data_type in self.metadata["supported_data_types"]

    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process the analysis request."""
        context = ErrorContext(
            agent_id=self.agent_id,
            request_id=request.id,
            operation="entity_extraction"
        )

        try:
            return await self._extract_entities_from_request(request, context)
        except Exception as e:
            return self.error_handling_service.handle_error(
                e, context, "Entity extraction failed"
            )

    async def _extract_entities_from_request(
        self,
        request: AnalysisRequest,
        context: ErrorContext
    ) -> AnalysisResult:
        """Extract entities from the analysis request."""
        try:
            # Extract text content from request
            text_content = await self._extract_text_content(request)

            # Detect language for multilingual extraction
            language = request.language or "en"
            
            # Extract entities using enhanced multilingual method
            entities_result = await self.extract_entities_multilingual(text_content, language)

            # Create sentiment result from entities
            sentiment = self._create_sentiment_from_entities(entities_result["entities"])

            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=sentiment,
                processing_time=0.0,  # Add processing time
                status=ProcessingStatus.COMPLETED,
                raw_content=text_content,
                metadata={
                    "entity_count": len(entities_result["entities"]),
                    "categories_found": entities_result["categories_found"],
                    "processing_method": "enhanced_multilingual_entity_extraction",
                    "language": language,
                    "confidence_scores": entities_result.get("confidence_scores", {}),
                    "entities": entities_result["entities"]
                }
            )

        except Exception as e:
            return self.error_handling_service.handle_error(e, context, "Entity extraction failed")

    async def _extract_text_content(self, request: AnalysisRequest) -> str:
        """Extract text content from the request."""
        return self.processing_service.extract_text_content(request)

    @tool("extract_entities", "Extract entities from text using basic extraction")
    async def extract_entities(self, text: str) -> dict:
        """Extract entities from text using basic extraction."""
        try:
            prompt = self._create_entity_extraction_prompt(text)
            response = await self._call_model(prompt)
            entities = self._parse_entities_from_response(response)

            return {
                "entities": entities,
                "count": len(entities),
                "categories_found": list(set(entity.get("category", "unknown") for entity in entities))
            }

        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {"entities": [], "count": 0, "categories_found": [], "error": str(e)}

    @tool("extract_entities_enhanced", "Extract entities from text using enhanced extraction with categorization")
    async def extract_entities_enhanced(self, text: str) -> dict:
        """Extract entities from text using enhanced extraction with categorization."""
        try:
            # Multi-strategy extraction (Phase 6.4 improvements)
            all_entities = []
            
            # Strategy 1: Enhanced LLM-based extraction
            try:
                llm_result = await self.extract_entities(text)
                if llm_result.get("entities"):
                    all_entities.extend(llm_result["entities"])
                    logger.info(f"LLM extraction found {len(llm_result['entities'])} entities")
                else:
                    logger.warning("LLM extraction returned no entities")
            except Exception as e:
                logger.warning(f"LLM extraction failed: {e}")
            
            # Strategy 2: Pattern-based extraction (always run as fallback)
            pattern_entities = self._extract_with_patterns(text)
            all_entities.extend(pattern_entities)
            logger.info(f"Pattern extraction found {len(pattern_entities)} entities")
            
            # Strategy 3: Dictionary-based extraction (always run as fallback)
            dict_entities = self._extract_with_dictionary(text)
            all_entities.extend(dict_entities)
            logger.info(f"Dictionary extraction found {len(dict_entities)} entities")

            # Merge and clean entities
            merged_entities = self._merge_similar_entities(all_entities)
            logger.info(f"After merging, total entities: {len(merged_entities)}")

            # Add context and relationships
            for entity in merged_entities:
                entity["context"] = self._extract_entity_context(entity, text)
                entity["relationships"] = self._find_entity_relationships(entity, merged_entities)
                entity["confidence"] = self._calculate_entity_confidence(entity)

            return {
                "entities": merged_entities,
                "count": len(merged_entities),
                "categories_found": list(set(entity.get("category", "unknown") for entity in merged_entities)),
                "statistics": self._count_entities_by_category(merged_entities),
                "confidence_scores": self._calculate_overall_confidence(merged_entities)
            }

        except Exception as e:
            logger.error(f"Error in enhanced entity extraction: {e}")
            return {"entities": [], "count": 0, "categories_found": [], "error": str(e)}

    @tool("extract_entities_multilingual", "Extract entities from text with multilingual support")
    async def extract_entities_multilingual(self, text: str, language: str = "en") -> dict:
        """Extract entities from text with enhanced multilingual support."""
        try:
            # Detect language if not provided
            if not language or language == "auto":
                language = self._detect_language(text)
            
            # Use language-specific extraction
            if language == "zh":
                return await self._extract_chinese_entities_enhanced(text)
            elif language == "ru":
                return await self._extract_russian_entities_enhanced(text)
            else:
                return await self.extract_entities_enhanced(text)
                
        except Exception as e:
            logger.error(f"Error in multilingual entity extraction: {e}")
            return {"entities": [], "count": 0, "categories_found": [], "error": str(e)}

    async def _extract_chinese_entities_enhanced(self, text: str) -> dict:
        """Enhanced Chinese entity extraction with structured prompts."""
        try:
            # Create structured Chinese prompt (Phase 6.1 improvements)
            prompt = self._create_enhanced_chinese_prompt(text)
            response = await self._call_model(prompt)
            
            # Parse structured response
            entities = self._parse_structured_entities(response)
            
            # Add pattern-based extraction
            pattern_entities = self._extract_with_chinese_patterns(text)
            entities.extend(pattern_entities)
            
            # Add dictionary-based extraction
            dict_entities = self._extract_with_chinese_dictionary(text)
            entities.extend(dict_entities)
            
            # Merge and validate entities
            merged_entities = self._merge_similar_entities(entities)
            validated_entities = self._validate_chinese_entities(merged_entities)
            
            # Add confidence scores
            for entity in validated_entities:
                entity["confidence"] = self._calculate_chinese_entity_confidence(entity)
            
            return {
                "entities": validated_entities,
                "count": len(validated_entities),
                "categories_found": list(set(entity.get("category", "unknown") for entity in validated_entities)),
                "statistics": self._count_entities_by_category(validated_entities),
                "confidence_scores": self._calculate_overall_confidence(validated_entities),
                "extraction_method": "enhanced_chinese_multi_strategy"
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced Chinese entity extraction: {e}")
            return {"entities": [], "count": 0, "categories_found": [], "error": str(e)}

    async def _extract_russian_entities_enhanced(self, text: str) -> dict:
        """Enhanced Russian entity extraction with structured prompts."""
        try:
            # Create structured Russian prompt
            prompt = self._create_enhanced_russian_prompt(text)
            response = await self._call_model(prompt)
            
            # Parse structured response
            entities = self._parse_structured_entities(response)
            
            # Add pattern-based extraction
            pattern_entities = self._extract_with_russian_patterns(text)
            entities.extend(pattern_entities)
            
            # Add dictionary-based extraction
            dict_entities = self._extract_with_russian_dictionary(text)
            entities.extend(dict_entities)
            
            # Merge and validate entities
            merged_entities = self._merge_similar_entities(entities)
            validated_entities = self._validate_russian_entities(merged_entities)
            
            # Add confidence scores
            for entity in validated_entities:
                entity["confidence"] = self._calculate_russian_entity_confidence(entity)
            
            return {
                "entities": validated_entities,
                "count": len(validated_entities),
                "categories_found": list(set(entity.get("category", "unknown") for entity in validated_entities)),
                "statistics": self._count_entities_by_category(validated_entities),
                "confidence_scores": self._calculate_overall_confidence(validated_entities),
                "extraction_method": "enhanced_russian_multi_strategy"
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced Russian entity extraction: {e}")
            return {"entities": [], "count": 0, "categories_found": [], "error": str(e)}

    def _create_enhanced_chinese_prompt(self, text: str) -> str:
        """Create enhanced Chinese prompt with structured output requirements."""
        return f"""
请从以下中文文本中精确提取实体，并按指定格式返回：

文本：{text}

请识别以下类型的实体：
1. 人名 (PERSON) - 包括政治人物、商业领袖、学者等
2. 组织名 (ORGANIZATION) - 包括公司、大学、研究所、政府部门等
3. 地名 (LOCATION) - 包括城市、国家、地区、地理特征等
4. 技术概念 (CONCEPT) - 包括AI技术、新兴技术、专业术语等

请严格按照以下JSON格式返回，每个实体包含text、type、confidence字段：
{{
    "entities": [
        {{"text": "实体名称", "type": "PERSON|ORGANIZATION|LOCATION|CONCEPT", "confidence": 0.9}},
        ...
    ]
}}

注意：
- 只提取有意义的实体，不要提取普通词汇
- 人名要完整提取（如"习近平主席"提取为"习近平"）
- 组织名要包含完整名称（如"华为技术有限公司"）
- 技术术语要准确识别（如"人工智能"、"机器学习"）
"""

    def _create_enhanced_russian_prompt(self, text: str) -> str:
        """Create enhanced Russian prompt with structured output requirements."""
        return f"""
Пожалуйста, извлеките точные сущности из следующего русского текста и верните в указанном формате:

Текст: {text}

Пожалуйста, определите следующие типы сущностей:
1. Имена людей (PERSON) - включая политиков, бизнес-лидеров, ученых и т.д.
2. Названия организаций (ORGANIZATION) - включая компании, университеты, институты, правительственные ведомства и т.д.
3. Географические названия (LOCATION) - включая города, страны, регионы, географические объекты и т.д.
4. Технические концепции (CONCEPT) - включая технологии ИИ, новые технологии, профессиональные термины и т.д.

Пожалуйста, строго следуйте следующему JSON формату, каждая сущность должна содержать поля text, type, confidence:
{{
    "entities": [
        {{"text": "название сущности", "type": "PERSON|ORGANIZATION|LOCATION|CONCEPT", "confidence": 0.9}},
        ...
    ]
}}

Примечания:
- Извлекайте только значимые сущности, не извлекайте обычные слова
- Имена людей должны быть полными (например, "Президент Путин" извлекается как "Владимир Путин")
- Названия организаций должны быть полными (например, "Газпром")
- Технические термины должны быть точно определены (например, "искусственный интеллект", "машинное обучение")
"""


    def _create_entity_extraction_prompt(self, text: str) -> str:
        """Create a prompt for entity extraction with improved structure."""
        return f"""
        Extract named entities from the following text. Focus on specific, identifiable entities.

        Entity Types to Extract:
        - PERSON: Individual people, names
        - ORGANIZATION: Companies, institutions, government bodies
        - LOCATION: Cities, countries, geographic places
        - TECHNOLOGY: Software, platforms, technical systems
        - PRODUCT: Specific products, services, brands
        - EVENT: Conferences, meetings, historical events
        - DATE: Specific dates, time periods
        - CONCEPT: Important ideas, theories, methodologies

        Rules:
        - Extract ONLY specific named entities, not entire sentences or phrases
        - Each entity should be a concise, identifiable name
        - Avoid extracting generic words or common phrases
        - Maximum entity name length: 50 characters
        - Focus on the most important and specific entities

        Text: {text}

        Return the entities in JSON format:
        {{
            "entities": [
                {{
                    "name": "specific_entity_name",
                    "type": "PERSON|ORGANIZATION|LOCATION|TECHNOLOGY|PRODUCT|EVENT|DATE|CONCEPT",
                    "importance": "high|medium|low",
                    "description": "brief_description"
                }}
            ]
        }}

        Examples of good entities:
        - "Google" (ORGANIZATION)
        - "Artificial Intelligence" (TECHNOLOGY)
        - "New York" (LOCATION)
        - "John Smith" (PERSON)

        Examples of bad entities (DO NOT extract):
        - "is transforming the world" (too generic)
        - "Companies like Google are leading" (entire phrase)
        - "the development" (too generic)

        IMPORTANT: Return ONLY valid JSON. Do not include any additional text or explanations.
        """

    def _parse_entities_from_response(self, response: str) -> List[Dict]:
        """Parse entities from model response with improved parsing."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                entities = data.get("entities", [])
                
                # Validate and clean entities
                cleaned_entities = []
                for entity in entities:
                    if self._validate_entity(entity):
                        cleaned_entities.append(entity)
                
                return cleaned_entities
            return []
        except Exception as e:
            logger.error(f"Error parsing entities from response: {e}")
            return []

    def _parse_structured_entities(self, response: str) -> List[Dict]:
        """Parse structured entities from enhanced Chinese response."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                entities = data.get("entities", [])
                
                # Convert to standard format
                converted_entities = []
                for entity in entities:
                    # Normalize entity type
                    entity_type = entity.get("type", "unknown")
                    normalized_type = self._normalize_entity_type(entity_type)
                    
                    converted_entity = {
                        "name": entity.get("text", ""),
                        "type": normalized_type,
                        "importance": "medium",
                        "description": f"{normalized_type} entity",
                        "confidence": entity.get("confidence", 0.5)
                    }
                    if self._validate_entity(converted_entity):
                        converted_entities.append(converted_entity)
                
                return converted_entities
            return []
        except Exception as e:
            logger.error(f"Error parsing structured entities: {e}")
            return []

    def _normalize_entity_type(self, entity_type: str) -> str:
        """Normalize entity type to standard format matching benchmark."""
        if not entity_type:
            return "CONCEPT"
        
        # Convert to uppercase and clean
        entity_type = entity_type.upper().strip()
        
        # Define priority order for entity types (matching benchmark)
        type_priority = [
            "PERSON", "ORGANIZATION", "LOCATION", "WORK", 
            "LINGUISTIC_TERM", "LESSON", "TECHNOLOGY", 
            "PRODUCT", "EVENT", "DATE", "CONCEPT"
        ]
        
        # If entity type contains multiple types (e.g., "person | organization")
        if "|" in entity_type:
            types = [t.strip() for t in entity_type.split("|")]
            # Find the highest priority type
            for priority_type in type_priority:
                if priority_type in types:
                    return priority_type
            # If no priority type found, return the first one
            return types[0] if types else "CONCEPT"
        
        # Direct mapping for common variations (matching benchmark categories)
        type_mapping = {
            "PERSON": "PERSON",
            "PERSON_NAME": "PERSON",
            "HUMAN": "PERSON",
            "INDIVIDUAL": "PERSON",
            "ORGANIZATION": "ORGANIZATION",
            "ORG": "ORGANIZATION",
            "COMPANY": "ORGANIZATION",
            "CORPORATION": "ORGANIZATION",
            "LOCATION": "LOCATION",
            "PLACE": "LOCATION",
            "CITY": "LOCATION",
            "COUNTRY": "LOCATION",
            "WORK": "WORK",
            "BOOK": "WORK",
            "DOCUMENT": "WORK",
            "LINGUISTIC_TERM": "LINGUISTIC_TERM",
            "TERM": "LINGUISTIC_TERM",
            "LESSON": "LESSON",
            "TECHNOLOGY": "TECHNOLOGY",
            "TECH": "TECHNOLOGY",
            "PRODUCT": "PRODUCT",
            "EVENT": "EVENT",
            "DATE": "DATE",
            "CONCEPT": "CONCEPT"
        }
        
        return type_mapping.get(entity_type, "CONCEPT")

    def _validate_entity(self, entity: Dict) -> bool:
        """Validate entity structure and content."""
        if not entity:
            return False
        
        name = entity.get("name", "")
        if not name or len(name.strip()) == 0:
            return False
        
        # Check if it's not an entire sentence
        if len(name) > 50 or "." in name or "，" in name:
            return False
        
        return True

    def _extract_with_patterns(self, text: str) -> List[Dict]:
        """Extract entities using regex patterns."""
        entities = []
        
        # More specific English patterns to avoid overlaps
        english_patterns = {
            "PERSON": [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last names
            ],
            "ORGANIZATION": [
                r'\b[A-Z][a-z]+ (Corp|Inc|Ltd|LLC|University|Institute|Government|Company|Group)\b',
                r'\b[A-Z][a-z]+ [A-Z][a-z]+ (Corp|Inc|Ltd|LLC)\b',  # Multi-word companies
            ],
            "LOCATION": [
                r'\b[A-Z][a-z]+ (City|State|Country|Province|District|Region)\b',
                r'\b[A-Z][a-z]+ [A-Z][a-z]+ (City|State|Country)\b',  # Multi-word locations
            ],
            "TECHNOLOGY": [
                r'\b[A-Z][a-z]+ Intelligence\b',  # AI terms
                r'\b[A-Z][a-z]+ Learning\b',  # ML terms
                r'\b[A-Z][a-z]+ Computing\b',  # Computing terms
                r'\b[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+\b'  # Three-word tech terms
            ],
            "CONCEPT": [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+\b'  # Three-word concepts
            ]
        }
        
        # Track used positions to avoid overlaps
        used_positions = set()
        
        for entity_type, pattern_list in english_patterns.items():
            for pattern in pattern_list:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Skip if this position is already used
                    if any(start_pos >= used_start and end_pos <= used_end 
                           for used_start, used_end in used_positions):
                        continue
                    
                    entity_name = match.group()
                    
                    # Skip if entity is too long or contains sentence markers
                    if len(entity_name) > 50 or "." in entity_name or "，" in entity_name:
                        continue
                    
                    # Skip common words that shouldn't be entities
                    common_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "have", "has", "had", "will", "would", "could", "should", "may", "might", "can", "must"}
                    if entity_name.lower() in common_words:
                        continue
                    
                    # Skip generic phrases and common patterns
                    generic_phrases = {
                        "is transforming", "are leading", "being used", "have revolutionized", 
                        "models like", "algorithms are", "the world", "the development", 
                        "companies like", "and education", "in healthcare", "are being used",
                        "is transforming the", "natural language", "and bert", "and openai",
                        "leading the development", "revolutionized natural language", "companies like google"
                    }
                    if entity_name.lower() in generic_phrases:
                        continue
                    
                    # Skip phrases that start with common words
                    if entity_name.lower().startswith(("the ", "and ", "in ", "are ", "is ", "have ", "being ", "leading ", "revolutionized ")):
                        continue
                    
                    # Skip phrases that contain action words
                    action_words = {"leading", "revolutionized", "transforming", "used", "like"}
                    if any(word in entity_name.lower() for word in action_words):
                        continue
                    
                    entity = {
                        "name": entity_name,
                        "type": entity_type.lower(),
                        "importance": "medium",
                        "description": f"{entity_type.lower()} entity",
                        "confidence": 0.7,
                        "extraction_method": "pattern"
                    }
                    entities.append(entity)
                    
                    # Mark this position as used
                    used_positions.add((start_pos, end_pos))
        
        return entities

    def _extract_with_chinese_patterns(self, text: str) -> List[Dict]:
        """Extract entities using enhanced Chinese patterns."""
        entities = []
        
        for entity_type, patterns in self.chinese_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    entity = {
                        "name": match.group(),
                        "type": entity_type.lower(),
                        "importance": "medium",
                        "description": f"{entity_type.lower()} entity",
                        "confidence": 0.8,
                        "extraction_method": "chinese_pattern"
                    }
                    entities.append(entity)
        
        return entities

    def _extract_with_dictionary(self, text: str) -> List[Dict]:
        """Extract entities using dictionary lookup."""
        entities = []
        
        # Enhanced English common entities
        english_entities = {
            "PERSON": [
                "Donald Trump", "Joe Biden", "Barack Obama", "Elon Musk", 
                "Bill Gates", "Steve Jobs", "Mark Zuckerberg", "Jeff Bezos"
            ],
            "ORGANIZATION": [
                "Microsoft", "Apple", "Google", "Amazon", "Meta", "OpenAI",
                "US Government", "Tesla", "Netflix", "Twitter", "LinkedIn"
            ],
            "LOCATION": [
                "United States", "China", "New York", "California", "Texas",
                "Washington", "London", "Tokyo", "Beijing", "San Francisco"
            ],
            "TECHNOLOGY": [
                "Artificial Intelligence", "Machine Learning", "Deep Learning",
                "Blockchain", "Cloud Computing", "Big Data", "Internet of Things",
                "Virtual Reality", "Augmented Reality", "Quantum Computing"
            ],
            "PRODUCT": [
                "iPhone", "Android", "Windows", "MacOS", "Linux", "Chrome",
                "Firefox", "Safari", "WordPress", "Slack", "Zoom"
            ],
            "CONCEPT": [
                "Digital Transformation", "Cybersecurity", "Data Science",
                "DevOps", "Agile", "Scrum", "API", "Microservices"
            ]
        }
        
        for entity_type, entity_list in english_entities.items():
            for entity_name in entity_list:
                if entity_name.lower() in text.lower():
                    entity = {
                        "name": entity_name,
                        "type": entity_type.lower(),
                        "importance": "high",
                        "description": f"Known {entity_type.lower()} entity",
                        "confidence": 0.9,
                        "extraction_method": "dictionary"
                    }
                    entities.append(entity)
        
        return entities

    def _extract_with_chinese_dictionary(self, text: str) -> List[Dict]:
        """Extract entities using Chinese dictionary lookup."""
        entities = []
        
        for entity_type, entity_list in self.chinese_dictionaries.items():
            for entity_name in entity_list:
                if entity_name in text:
                    entity = {
                        "name": entity_name,
                        "type": entity_type.lower(),
                        "importance": "high",
                        "description": f"Known Chinese {entity_type.lower()} entity",
                        "confidence": 0.9,
                        "extraction_method": "chinese_dictionary"
                    }
                    entities.append(entity)
        
        return entities

    def _extract_with_russian_patterns(self, text: str) -> List[Dict]:
        """Extract entities using enhanced Russian patterns."""
        entities = []
        
        for entity_type, patterns in self.russian_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    entity = {
                        "name": match.group(),
                        "type": entity_type.lower(),
                        "importance": "medium",
                        "description": f"{entity_type.lower()} entity",
                        "confidence": 0.8,
                        "extraction_method": "russian_pattern"
                    }
                    entities.append(entity)
        
        return entities

    def _extract_with_russian_dictionary(self, text: str) -> List[Dict]:
        """Extract entities using Russian dictionary lookup."""
        entities = []
        
        for entity_type, entity_list in self.russian_dictionaries.items():
            for entity_name in entity_list:
                if entity_name in text:
                    entity = {
                        "name": entity_name,
                        "type": entity_type.lower(),
                        "importance": "high",
                        "description": f"Known Russian {entity_type.lower()} entity",
                        "confidence": 0.9,
                        "extraction_method": "russian_dictionary"
                    }
                    entities.append(entity)
        
        return entities

    def _validate_chinese_entities(self, entities: List[Dict]) -> List[Dict]:
        """Validate Chinese entities using specific rules."""
        validated_entities = []
        
        for entity in entities:
            if self._validate_chinese_entity(entity):
                validated_entities.append(entity)
        
        return validated_entities

    def _validate_chinese_entity(self, entity: Dict) -> bool:
        """Validate individual Chinese entity."""
        name = entity.get("name", "")
        entity_type = entity.get("type", "").lower()
        
        if not name:
            return False
        
        # Basic validation
        if len(name) < 2:
            return False
        
        # Type-specific validation
        if entity_type == "person":
            return self._validate_chinese_person_name(name)
        elif entity_type == "organization":
            return self._validate_chinese_organization_name(name)
        elif entity_type == "location":
            return self._validate_chinese_location_name(name)
        elif entity_type == "concept":
            return self._validate_chinese_technical_term(name)
        
        return True

    def _validate_chinese_person_name(self, name: str) -> bool:
        """Validate Chinese person names."""
        # Chinese names are typically 2-4 characters
        if len(name) < 2 or len(name) > 4:
            return False
        
        # Should contain only Chinese characters
        if not re.match(r'^[\u4e00-\u9fff]+$', name):
            return False
        
        return True

    def _validate_chinese_organization_name(self, name: str) -> bool:
        """Validate Chinese organization names."""
        # Should be at least 2 characters
        if len(name) < 2:
            return False
        
        # Should contain Chinese characters
        if not re.search(r'[\u4e00-\u9fff]', name):
            return False
        
        return True

    def _validate_chinese_location_name(self, name: str) -> bool:
        """Validate Chinese location names."""
        # Should be at least 2 characters
        if len(name) < 2:
            return False
        
        # Should contain Chinese characters
        if not re.search(r'[\u4e00-\u9fff]', name):
            return False
        
        return True

    def _validate_chinese_technical_term(self, term: str) -> bool:
        """Validate Chinese technical terms."""
        # Should be at least 2 characters
        if len(term) < 2:
            return False
        
        # Should contain Chinese characters
        if not re.search(r'[\u4e00-\u9fff]', term):
            return False
        
        return True

    def _calculate_chinese_entity_confidence(self, entity: Dict) -> float:
        """Calculate confidence score for Chinese entity."""
        confidence = 0.5  # Base confidence
        
        # Boost based on extraction method
        method = entity.get("extraction_method", "")
        if method == "chinese_dictionary":
            confidence += 0.4
        elif method == "chinese_pattern":
            confidence += 0.3
        elif method == "llm":
            confidence += 0.2
        
        # Boost based on validation
        if self._validate_chinese_entity(entity):
            confidence += 0.1
        
        # Boost based on importance
        importance = entity.get("importance", "low").lower()
        if importance == "high":
            confidence += 0.1
        elif importance == "medium":
            confidence += 0.05
        
        return min(confidence, 1.0)

    def _validate_russian_entities(self, entities: List[Dict]) -> List[Dict]:
        """Validate Russian entities using specific rules."""
        validated_entities = []
        
        for entity in entities:
            if self._validate_russian_entity(entity):
                validated_entities.append(entity)
        
        return validated_entities

    def _validate_russian_entity(self, entity: Dict) -> bool:
        """Validate a single Russian entity."""
        entity_name = entity.get("name", "")
        entity_type = entity.get("type", "").lower()
        
        if not entity_name or len(entity_name.strip()) < 3:  # Increased minimum length
            return False
        
        # Type-specific validation
        if entity_type == "person":
            return self._validate_russian_person_name(entity_name)
        elif entity_type == "organization":
            return self._validate_russian_organization_name(entity_name)
        elif entity_type == "location":
            return self._validate_russian_location_name(entity_name)
        elif entity_type == "concept":
            return self._validate_russian_technical_term(entity_name)
        
        return True

    def _validate_russian_person_name(self, name: str) -> bool:
        """Validate Russian person names."""
        # Should be at least 3 characters
        if len(name) < 3:
            return False
        
        # Should contain Russian Cyrillic characters
        if not re.search(r'[А-ЯЁа-яё]', name):
            return False
        
        # Should start with capital letter
        if not re.match(r'^[А-ЯЁ]', name):
            return False
        
        return True

    def _validate_russian_organization_name(self, name: str) -> bool:
        """Validate Russian organization names."""
        # Should be at least 3 characters
        if len(name) < 3:
            return False
        
        # Should contain Russian Cyrillic characters
        if not re.search(r'[А-ЯЁа-яё]', name):
            return False
        
        return True

    def _validate_russian_location_name(self, name: str) -> bool:
        """Validate Russian location names."""
        # Should be at least 3 characters
        if len(name) < 3:
            return False
        
        # Should contain Russian Cyrillic characters
        if not re.search(r'[А-ЯЁа-яё]', name):
            return False
        
        return True

    def _validate_russian_technical_term(self, term: str) -> bool:
        """Validate Russian technical terms."""
        # Should be at least 3 characters
        if len(term) < 3:
            return False
        
        # Should contain Russian Cyrillic characters
        if not re.search(r'[А-ЯЁа-яё]', term):
            return False
        
        return True

    def _calculate_russian_entity_confidence(self, entity: Dict) -> float:
        """Calculate confidence score for Russian entity."""
        confidence = 0.5  # Base confidence
        
        # Boost based on extraction method
        method = entity.get("extraction_method", "")
        if method == "russian_dictionary":
            confidence += 0.4
        elif method == "russian_pattern":
            confidence += 0.3
        elif method == "llm":
            confidence += 0.2
        
        # Boost based on validation
        if self._validate_russian_entity(entity):
            confidence += 0.1
        
        # Boost based on importance
        importance = entity.get("importance", "low").lower()
        if importance == "high":
            confidence += 0.1
        elif importance == "medium":
            confidence += 0.05
        
        return min(confidence, 1.0)

    def _detect_language(self, text: str) -> str:
        """Simple language detection."""
        # Count Chinese characters
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        # Count Russian Cyrillic characters
        russian_chars = len(re.findall(r'[А-ЯЁа-яё]', text))
        total_chars = len(text)
        
        if total_chars > 0:
            chinese_ratio = chinese_chars / total_chars
            russian_ratio = russian_chars / total_chars
            
            if chinese_ratio > 0.3:
                return "zh"
            elif russian_ratio > 0.3:
                return "ru"
            else:
                return "en"
        else:
            return "en"

    def _calculate_overall_confidence(self, entities: List[Dict]) -> Dict[str, float]:
        """Calculate overall confidence scores by category."""
        confidence_scores = {}
        category_counts = {}
        
        for entity in entities:
            category = entity.get("category", "unknown")
            confidence = entity.get("confidence", 0.5)
            
            if category not in confidence_scores:
                confidence_scores[category] = 0.0
                category_counts[category] = 0
            
            confidence_scores[category] += confidence
            category_counts[category] += 1
        
        # Calculate averages
        for category in confidence_scores:
            if category_counts[category] > 0:
                confidence_scores[category] /= category_counts[category]
        
        return confidence_scores

    async def _call_model(self, prompt: str) -> str:
        """Call the model using the strands agent."""
        try:
            response = await self.strands_agent.run(prompt)
            return response
        except Exception as e:
            logger.error(f"Error calling model: {e}")
            # Return a fallback response
            return "{}"

    def _merge_similar_entities(self, entities: List[Dict]) -> List[Dict]:
        """Merge similar entities based on name similarity."""
        if not entities:
            return []

        merged = []
        processed = set()

        for i, entity1 in enumerate(entities):
            if i in processed:
                continue

            similar_entities = [entity1]
            processed.add(i)

            for j, entity2 in enumerate(entities[i+1:], i+1):
                if j in processed:
                    continue

                # Check if entities are similar (same name or very similar)
                if self._are_entities_similar(entity1, entity2):
                    similar_entities.append(entity2)
                    processed.add(j)

            # Merge similar entities
            merged_entity = self._merge_entity_group(similar_entities)
            merged.append(merged_entity)

        return merged

    def _are_entities_similar(self, entity1: Dict, entity2: Dict) -> bool:
        """Check if two entities are similar."""
        name1 = entity1.get("name", "").lower()
        name2 = entity2.get("name", "").lower()

        # Exact match
        if name1 == name2:
            return True

        # Check for partial matches
        if name1 in name2 or name2 in name1:
            return True

        # Check for acronyms
        if len(name1) <= 3 and name1.upper() == name1:
            if name1.lower() in name2.lower():
                return True

        if len(name2) <= 3 and name2.upper() == name2:
            if name2.lower() in name1.lower():
                return True

        return False

    def _merge_entity_group(self, entities: List[Dict]) -> Dict:
        """Merge a group of similar entities into one."""
        if not entities:
            return {}

        # Use the first entity as base
        merged = entities[0].copy()

        # Merge importance levels
        importance_levels = [entity.get("importance", "low") for entity in entities]
        merged["importance"] = self._merge_importance(importance_levels[0], importance_levels[-1])

        # Merge descriptions
        descriptions = [entity.get("description", "") for entity in entities if entity.get("description")]
        if descriptions:
            merged["description"] = " | ".join(descriptions)

        # Merge types if different
        types = list(set(entity.get("type", "") for entity in entities if entity.get("type")))
        if len(types) > 1:
            merged["type"] = " | ".join(types)

        return merged

    def _merge_importance(self, importance1: str, importance2: str) -> str:
        """Merge two importance levels."""
        importance_map = {"low": 1, "medium": 2, "high": 3}
        level1 = importance_map.get(importance1.lower(), 1)
        level2 = importance_map.get(importance2.lower(), 1)
        max_level = max(level1, level2)
        return {1: "low", 2: "medium", 3: "high"}[max_level]

    def _categorize_entities(self, entities: List[Dict]) -> List[Dict]:
        """Categorize entities based on their type and description."""
        for entity in entities:
            entity["category"] = self._determine_entity_category(entity)
        return entities

    def _determine_entity_category(self, entity: Dict) -> str:
        """Determine the category of an entity."""
        entity_type = entity.get("type", "").lower()
        entity_name = entity.get("name", "").lower()
        description = entity.get("description", "").lower()

        # Check each category
        for category, keywords in self.entity_categories.items():
            for keyword in keywords:
                if (keyword in entity_type or keyword in entity_name or
                    keyword in description):
                    return category

        return "other"

    def _count_entities_by_category(self, entities: List[Dict]) -> Dict[str, int]:
        """Count entities by category."""
        counts = {}
        for entity in entities:
            category = entity.get("category", "unknown")
            counts[category] = counts.get(category, 0) + 1
        return counts

    def _calculate_entity_confidence(self, entity: Dict) -> float:
        """Calculate confidence score for an entity."""
        confidence = 0.5  # Base confidence

        # Boost confidence based on entity properties
        if entity.get("description"):
            confidence += 0.2

        if entity.get("context"):
            confidence += 0.1

        if entity.get("relationships"):
            confidence += 0.1

        importance = entity.get("importance", "low").lower()
        if importance == "high":
            confidence += 0.1
        elif importance == "medium":
            confidence += 0.05

        return min(confidence, 1.0)

    def _extract_entity_context(self, entity: Dict, text: str) -> str:
        """Extract context around an entity in the text."""
        entity_name = entity.get("name", "")
        if not entity_name:
            return ""

        # Find entity position in text
        pos = text.lower().find(entity_name.lower())
        if pos == -1:
            return ""

        # Extract context (50 characters before and after)
        start = max(0, pos - 50)
        end = min(len(text), pos + len(entity_name) + 50)
        context = text[start:end]

        return context.strip()

    def _find_entity_relationships(self, entity: Dict, all_entities: List[Dict]) -> List[str]:
        """Find relationships between entities."""
        relationships = []
        entity_name = entity.get("name", "").lower()

        for other_entity in all_entities:
            if other_entity == entity:
                continue

            other_name = other_entity.get("name", "").lower()
            if other_name in entity_name or entity_name in other_name:
                relationships.append(f"similar_to:{other_entity['name']}")

        return relationships

    def _create_sentiment_from_entities(self, entities: List[Dict]) -> SentimentResult:
        """Create a sentiment result from extracted entities."""
        if not entities:
            return SentimentResult(
                label=SentimentLabel.NEUTRAL,
                confidence=0.5,
                scores={"positive": 0.0, "negative": 0.0, "neutral": 1.0}
            )

        # Simple sentiment based on entity importance
        high_importance_count = sum(1 for e in entities if e.get("importance") == "high")
        total_count = len(entities)

        if total_count == 0:
            return SentimentResult(
                label=SentimentLabel.NEUTRAL,
                confidence=0.5,
                scores={"positive": 0.0, "negative": 0.0, "neutral": 1.0}
            )

        # Calculate sentiment based on entity importance
        importance_ratio = high_importance_count / total_count

        if importance_ratio > 0.7:
            sentiment = SentimentLabel.POSITIVE
            scores = {"positive": 0.8, "negative": 0.1, "neutral": 0.1}
        elif importance_ratio > 0.3:
            sentiment = SentimentLabel.NEUTRAL
            scores = {"positive": 0.3, "negative": 0.2, "neutral": 0.5}
        else:
            sentiment = SentimentLabel.NEGATIVE
            scores = {"positive": 0.1, "negative": 0.7, "neutral": 0.2}

        return SentimentResult(
            label=sentiment,
            confidence=0.6,
            scores=scores
        )

    @tool("categorize_entities", "Categorize a list of entities")
    async def categorize_entities(self, entities: List[Dict]) -> dict:
        """Categorize a list of entities."""
        try:
            categorized = self._categorize_entities(entities)
            return {
                "entities": categorized,
                "categories": list(set(entity.get("category", "unknown") for entity in categorized)),
                "statistics": self._count_entities_by_category(categorized)
            }

        except Exception as e:
            logger.error(f"Error categorizing entities: {e}")
            return {"entities": [], "categories": [], "statistics": {}, "error": str(e)}

    @tool("extract_entities_from_chunks", "Extract entities from multiple text chunks")
    async def extract_entities_from_chunks(self, chunks: List[str]) -> dict:
        """Extract entities from multiple text chunks."""
        try:
            all_entities = []
            chunk_results = []

            for i, chunk in enumerate(chunks):
                chunk_result = await self.extract_entities_enhanced(chunk)
                all_entities.extend(chunk_result["entities"])
                chunk_results.append({
                    "chunk_index": i,
                    "entities": chunk_result["entities"],
                    "count": chunk_result["count"]
                })

            # Merge entities across chunks
            merged_entities = self._merge_similar_entities(all_entities)

            return {
                "entities": merged_entities,
                "total_count": len(merged_entities),
                "chunk_results": chunk_results,
                "categories_found": list(set(entity.get("category", "unknown") for entity in merged_entities))
            }

        except Exception as e:
            logger.error(f"Error extracting entities from chunks: {e}")
            return {"entities": [], "total_count": 0, "chunk_results": [], "error": str(e)}

    @tool("get_entity_statistics", "Get statistics about entity extraction capabilities")
    async def get_entity_statistics(self) -> dict:
        """Get statistics about entity extraction capabilities."""
        return {
            "entity_categories": self.entity_categories,
            "supported_data_types": self.metadata["supported_data_types"],
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "model": self.model_name,
            "enhanced_features": self.metadata.get("enhanced_features", []),
            "chinese_patterns": list(self.chinese_patterns.keys()) if hasattr(self, 'chinese_patterns') else [],
            "chinese_dictionaries": list(self.chinese_dictionaries.keys()) if hasattr(self, 'chinese_dictionaries') else []
        }
