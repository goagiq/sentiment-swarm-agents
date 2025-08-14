"""
Entity types configuration for multilingual processing.
Supports different entity types for different languages and cultures.
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum


class EntityType(Enum):
    """Standard entity types supported across languages."""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    EVENT = "EVENT"
    CONCEPT = "CONCEPT"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    QUANTITY = "QUANTITY"
    MISC = "MISC"


@dataclass
class LanguageEntityConfig:
    """Entity configuration for a specific language."""
    language_code: str
    language_name: str
    supported_entity_types: List[str]
    entity_patterns: Dict[str, List[str]]
    confidence_thresholds: Dict[str, float]
    extraction_settings: Dict[str, Any]


class EntityTypesConfig:
    """Configuration manager for entity types across languages."""
    
    def __init__(self):
        self.language_configs: Dict[str, LanguageEntityConfig] = {}
        self._initialize_default_configs()
    
    def _initialize_default_configs(self):
        """Initialize default configurations for supported languages."""
        
        # English configuration
        self.language_configs["en"] = LanguageEntityConfig(
            language_code="en",
            language_name="English",
            supported_entity_types=[
                "PERSON", "ORGANIZATION", "LOCATION", "EVENT", 
                "CONCEPT", "DATE", "TIME", "MONEY", "PERCENT", "QUANTITY", "MISC"
            ],
            entity_patterns={
                "PERSON": ["Mr.", "Ms.", "Dr.", "Prof.", "President", "CEO", "Director"],
                "ORGANIZATION": ["Inc.", "Corp.", "LLC", "Ltd.", "Company", "University", "Institute"],
                "LOCATION": ["Street", "Avenue", "Road", "City", "State", "Country", "River", "Mountain"],
                "EVENT": ["Conference", "Meeting", "Summit", "Election", "War", "Treaty"],
                "CONCEPT": ["Policy", "Strategy", "Technology", "Innovation", "Research"],
                "DATE": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
                "TIME": ["AM", "PM", "o'clock", "hour", "minute", "second"],
                "MONEY": ["dollar", "euro", "pound", "yen", "yuan", "$", "€", "£", "¥"],
                "PERCENT": ["percent", "%", "percentage"],
                "QUANTITY": ["million", "billion", "thousand", "hundred", "dozen", "pair"],
                "MISC": ["etc.", "e.g.", "i.e.", "vs.", "etc"]
            },
            confidence_thresholds={
                "PERSON": 0.8,
                "ORGANIZATION": 0.7,
                "LOCATION": 0.8,
                "EVENT": 0.6,
                "CONCEPT": 0.5,
                "DATE": 0.9,
                "TIME": 0.8,
                "MONEY": 0.9,
                "PERCENT": 0.9,
                "QUANTITY": 0.7,
                "MISC": 0.4
            },
            extraction_settings={
                "min_entity_length": 2,
                "max_entity_length": 50,
                "use_enhanced_extraction": True,
                "use_regex_patterns": True,
                "use_llm_extraction": True
            }
        )
        
        # Chinese configuration
        self.language_configs["zh"] = LanguageEntityConfig(
            language_code="zh",
            language_name="Chinese",
            supported_entity_types=[
                "PERSON", "ORGANIZATION", "LOCATION", "EVENT", 
                "CONCEPT", "DATE", "TIME", "MONEY", "PERCENT", "QUANTITY", "MISC"
            ],
            entity_patterns={
                "PERSON": ["先生", "女士", "博士", "教授", "总统", "主席", "总理", "部长"],
                "ORGANIZATION": ["公司", "企业", "大学", "学院", "研究所", "政府", "部门", "机构"],
                "LOCATION": ["省", "市", "县", "区", "街道", "路", "山", "河", "湖", "海"],
                "EVENT": ["会议", "峰会", "选举", "战争", "条约", "协议", "活动"],
                "CONCEPT": ["政策", "战略", "技术", "创新", "研究", "发展", "经济"],
                "DATE": ["年", "月", "日", "星期", "周", "世纪", "年代"],
                "TIME": ["时", "分", "秒", "上午", "下午", "晚上", "凌晨"],
                "MONEY": ["元", "美元", "欧元", "英镑", "日元", "人民币"],
                "PERCENT": ["百分之", "%", "百分比"],
                "QUANTITY": ["万", "亿", "千", "百", "十", "个", "只", "条"],
                "MISC": ["等等", "例如", "即", "对", "与"]
            },
            confidence_thresholds={
                "PERSON": 0.8,
                "ORGANIZATION": 0.7,
                "LOCATION": 0.8,
                "EVENT": 0.6,
                "CONCEPT": 0.5,
                "DATE": 0.9,
                "TIME": 0.8,
                "MONEY": 0.9,
                "PERCENT": 0.9,
                "QUANTITY": 0.7,
                "MISC": 0.4
            },
            extraction_settings={
                "min_entity_length": 1,
                "max_entity_length": 20,
                "use_enhanced_extraction": True,
                "use_regex_patterns": True,
                "use_llm_extraction": True,
                "use_character_based_extraction": True
            }
        )
        
        # Russian configuration
        self.language_configs["ru"] = LanguageEntityConfig(
            language_code="ru",
            language_name="Russian",
            supported_entity_types=[
                "PERSON", "ORGANIZATION", "LOCATION", "EVENT", 
                "CONCEPT", "DATE", "TIME", "MONEY", "PERCENT", "QUANTITY", "MISC"
            ],
            entity_patterns={
                "PERSON": ["господин", "госпожа", "доктор", "профессор", "президент", "министр", "директор"],
                "ORGANIZATION": ["компания", "корпорация", "университет", "институт", "правительство", "министерство"],
                "LOCATION": ["город", "область", "край", "республика", "улица", "проспект", "река", "гора"],
                "EVENT": ["конференция", "встреча", "выборы", "война", "договор", "соглашение"],
                "CONCEPT": ["политика", "стратегия", "технология", "инновация", "исследование", "развитие"],
                "DATE": ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"],
                "TIME": ["час", "минута", "секунда", "утро", "день", "вечер", "ночь"],
                "MONEY": ["рубль", "доллар", "евро", "фунт", "иена", "юань"],
                "PERCENT": ["процент", "%", "процентов"],
                "QUANTITY": ["миллион", "миллиард", "тысяча", "сотня", "десяток", "пара"],
                "MISC": ["и т.д.", "например", "то есть", "против", "и"]
            },
            confidence_thresholds={
                "PERSON": 0.8,
                "ORGANIZATION": 0.7,
                "LOCATION": 0.8,
                "EVENT": 0.6,
                "CONCEPT": 0.5,
                "DATE": 0.9,
                "TIME": 0.8,
                "MONEY": 0.9,
                "PERCENT": 0.9,
                "QUANTITY": 0.7,
                "MISC": 0.4
            },
            extraction_settings={
                "min_entity_length": 2,
                "max_entity_length": 50,
                "use_enhanced_extraction": True,
                "use_regex_patterns": True,
                "use_llm_extraction": True,
                "use_cyrillic_patterns": True
            }
        )
    
    def get_language_config(self, language_code: str) -> LanguageEntityConfig:
        """Get configuration for a specific language."""
        return self.language_configs.get(language_code.lower(), self.language_configs["en"])
    
    def get_supported_entity_types(self, language_code: str) -> List[str]:
        """Get supported entity types for a language."""
        config = self.get_language_config(language_code)
        return config.supported_entity_types
    
    def validate_entity_types(self, entity_types: List[str], language_code: str) -> List[str]:
        """Validate and normalize entity types for a language."""
        if not entity_types:
            return self.get_supported_entity_types(language_code)
        
        supported_types = self.get_supported_entity_types(language_code)
        validated_types = []
        
        for entity_type in entity_types:
            normalized_type = entity_type.upper()
            if normalized_type in supported_types:
                validated_types.append(normalized_type)
            else:
                # Try to map common variations
                if normalized_type in ["PERSON", "PERSONS", "PEOPLE"]:
                    validated_types.append("PERSON")
                elif normalized_type in ["ORG", "ORGANIZATION", "ORGANIZATIONS"]:
                    validated_types.append("ORGANIZATION")
                elif normalized_type in ["LOC", "LOCATION", "LOCATIONS", "PLACE", "PLACES"]:
                    validated_types.append("LOCATION")
                elif normalized_type in ["EVENT", "EVENTS"]:
                    validated_types.append("EVENT")
                elif normalized_type in ["CONCEPT", "CONCEPTS", "TOPIC", "TOPICS"]:
                    validated_types.append("CONCEPT")
        
        # If no valid types found, return all supported types
        if not validated_types:
            return supported_types
        
        return validated_types
    
    def get_entity_patterns(self, language_code: str, entity_type: str = None) -> Dict[str, List[str]]:
        """Get entity patterns for a language and optionally specific entity type."""
        config = self.get_language_config(language_code)
        if entity_type:
            return {entity_type.upper(): config.entity_patterns.get(entity_type.upper(), [])}
        return config.entity_patterns
    
    def get_confidence_threshold(self, language_code: str, entity_type: str) -> float:
        """Get confidence threshold for an entity type in a language."""
        config = self.get_language_config(language_code)
        return config.confidence_thresholds.get(entity_type.upper(), 0.5)
    
    def get_extraction_settings(self, language_code: str) -> Dict[str, Any]:
        """Get extraction settings for a language."""
        config = self.get_language_config(language_code)
        return config.extraction_settings


# Global instance
entity_types_config = EntityTypesConfig()
