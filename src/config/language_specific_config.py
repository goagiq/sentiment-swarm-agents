"""
Language-specific configuration for entity extraction and processing.
This file stores language-specific patterns, regex rules, and processing settings.
"""

from typing import Dict, List, Any
from dataclasses import dataclass
import re


@dataclass
class LanguageSpecificConfig:
    """Configuration for a specific language."""
    language_code: str
    language_name: str
    entity_patterns: Dict[str, List[str]]
    regex_patterns: Dict[str, str]
    processing_settings: Dict[str, Any]
    detection_patterns: List[str]  # Add detection patterns for language identification


# Language-specific configuration for enhanced entity extraction and processing
# This file contains language-specific settings, patterns, and processing rules

# Enhanced language-specific configuration
LANGUAGE_CONFIG = {
    "en": {
        "use_enhanced_extraction": True,
        "patterns": {
            "PERSON": [
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last
                r'\b[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z]\.\b',  # First M. L.
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Middle Last
            ],
            "ORGANIZATION": [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+Inc\.|\s+Corp\.|\s+Ltd\.|\s+LLC|\s+Company|\s+Corporation)\b',
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+University|\s+College|\s+Institute|\s+School)\b',
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+Government|\s+Agency|\s+Department|\s+Ministry)\b',
            ],
            "LOCATION": [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+City|\s+Town|\s+Village|\s+County|\s+State|\s+Country)\b',
                r'\b(?:New York|Los Angeles|Chicago|Houston|Phoenix|Philadelphia|San Antonio|San Diego|Dallas|San Jose)\b',
            ],
            "CONCEPT": [
                r'\b(?:artificial intelligence|machine learning|deep learning|neural networks)\b',
                r'\b(?:blockchain|cloud computing|big data|internet of things|IoT)\b',
                r'\b(?:quantum computing|cybersecurity|data science|robotics)\b',
            ]
        },
        "dictionaries": {
            "PERSON": [
                "John Smith", "Jane Doe", "Michael Johnson", "Sarah Wilson",
                "David Brown", "Emily Davis", "Robert Miller", "Lisa Garcia"
            ],
            "ORGANIZATION": [
                "Microsoft", "Apple", "Google", "Amazon", "Facebook", "Tesla",
                "Harvard University", "MIT", "Stanford University", "NASA"
            ],
            "LOCATION": [
                "United States", "New York", "California", "Texas", "Florida",
                "Washington", "London", "Paris", "Tokyo", "Beijing"
            ],
            "CONCEPT": [
                "artificial intelligence", "machine learning", "deep learning",
                "neural networks", "blockchain", "cloud computing", "big data"
            ]
        },
        "relationship_patterns": [
            r'\b(?:works for|employed by|CEO of|president of|director of)\b',
            r'\b(?:located in|based in|headquartered in|situated in)\b',
            r'\b(?:studied at|graduated from|attended|enrolled at)\b',
            r'\b(?:developed|created|invented|discovered|founded)\b',
            r'\b(?:part of|member of|belongs to|affiliated with)\b'
        ],
        "min_entity_length": 2,
        "max_entity_length": 50,
        "confidence_threshold": 0.6
    },
    
    "zh": {
        "use_enhanced_extraction": True,
        "patterns": {
            "PERSON": [
                r'[\u4e00-\u9fff]{2,4}',  # Chinese names (2-4 characters)
                r'[\u4e00-\u9fff]{2,4}\s+[\u4e00-\u9fff]{2,4}',  # Full names
            ],
            "ORGANIZATION": [
                r'[\u4e00-\u9fff]+(?:公司|集团|企业|银行|大学|学院|研究所|研究院|政府|部门)',
                r'[\u4e00-\u9fff]+(?:科技|技术|信息|网络|互联网|电子|通信|金融|投资)',
            ],
            "LOCATION": [
                r'[\u4e00-\u9fff]+(?:市|省|区|县|州|国|地区|城市)',
                r'(?:北京|上海|广州|深圳|杭州|南京|武汉|成都|西安|重庆)',
            ],
            "CONCEPT": [
                r'(?:人工智能|机器学习|深度学习|神经网络|自然语言处理|计算机视觉)',
                r'(?:区块链|云计算|大数据|物联网|量子计算|网络安全)',
            ]
        },
        "dictionaries": {
            "PERSON": [
                "习近平", "李克强", "马云", "马化腾", "李彦宏", "任正非",
                "王健林", "许家印", "杨元庆", "雷军"
            ],
            "ORGANIZATION": [
                "阿里巴巴", "腾讯", "百度", "华为", "小米", "京东",
                "清华大学", "北京大学", "复旦大学", "上海交通大学"
            ],
            "LOCATION": [
                "中国", "北京", "上海", "广州", "深圳", "杭州",
                "南京", "武汉", "成都", "西安"
            ],
            "CONCEPT": [
                "人工智能", "机器学习", "深度学习", "神经网络",
                "自然语言处理", "计算机视觉", "量子计算", "区块链"
            ]
        },
        "relationship_patterns": [
            r'(?:在|于|就职于|工作于|担任|出任|领导|管理)',
            r'(?:位于|坐落于|地处|在|属于|隶属于)',
            r'(?:毕业于|就读于|在|学习于|师从)',
            r'(?:开发|创造|发明|发现|创立|建立)',
            r'(?:属于|是|成为|加入|参与|合作)'
        ],
        "min_entity_length": 2,
        "max_entity_length": 20,
        "confidence_threshold": 0.7
    },
    
    "ru": {
        "use_enhanced_extraction": True,
        "patterns": {
            "PERSON": [
                r'\b[А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,})?\b',  # Full names (3+ chars each)
                r'\b[А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.\b',  # Name with initials
                r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,})*\s+(?:господин|госпожа|доктор|профессор)\b',  # With titles
            ],
            "ORGANIZATION": [
                r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[а-яё]{2,})*(?:ООО|ОАО|ЗАО|ПАО|ГК|Корпорация|Компания)\b',
                r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[а-яё]{2,})*(?:Университет|Институт|Академия|Университет)\b',
                r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[а-яё]{2,})*(?:Правительство|Министерство|Агентство)\b',
            ],
            "LOCATION": [
                r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[а-яё]{2,})*(?:город|область|край|республика|район)\b',
                r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[а-яё]{2,})*(?:улица|проспект|переулок|площадь)\b',
                r'\b(?:Москва|Санкт-Петербург|Новосибирск|Екатеринбург|Казань|Россия)\b',  # Major cities
            ],
            "CONCEPT": [
                r'\b(?:искусственный интеллект|машинное обучение|глубокое обучение)\b',
                r'\b(?:блокчейн|облачные вычисления|большие данные|интернет вещей)\b',
                r'\b(?:цифровая экономика|умное производство|зеленая энергия)\b',
            ]
        },
        "dictionaries": {
            "PERSON": [
                "Владимир Путин", "Дмитрий Медведев", "Сергей Лавров", "Алексей Миллер",
                "Герман Греф", "Андрей Костин", "Олег Дерипаска", "Роман Абрамович",
                "Михаил Фридман", "Алишер Усманов", "Леонид Михельсон", "Вагит Алекперов"
            ],
            "ORGANIZATION": [
                "Газпром", "Сбербанк", "Роснефть", "Лукойл", "Норникель",
                "МГУ", "СПбГУ", "МФТИ", "РАН", "Сколково", "ВШЭ"
            ],
            "LOCATION": [
                "Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург", "Казань",
                "Россия", "США", "Китай", "Германия", "Франция", "Великобритания"
            ],
            "CONCEPT": [
                "искусственный интеллект", "машинное обучение", "глубокое обучение",
                "нейронные сети", "обработка естественного языка", "компьютерное зрение",
                "квантовые вычисления", "блокчейн", "облачные вычисления"
            ]
        },
        "relationship_patterns": [
            r'\b(?:работает в|работает на|возглавляет|руководит|является|находится в)\b',
            r'\b(?:расположен в|находится в|базируется в|штаб-квартира в)\b',
            r'\b(?:учился в|окончил|закончил|получил образование в)\b',
            r'\b(?:разработал|создал|изобрел|открыл|основал|учредил)\b',
            r'\b(?:является частью|входит в|принадлежит к|аффилирован с)\b'
        ],
        "min_entity_length": 3,  # Increased minimum length for Russian
        "max_entity_length": 50,
        "confidence_threshold": 0.7,
        "font_family": "Arial Unicode MS, DejaVu Sans, sans-serif",  # Better font support
        "encoding": "utf-8"
    }
}

def should_use_enhanced_extraction(language: str) -> bool:
    """Check if enhanced extraction should be used for a given language."""
    return LANGUAGE_CONFIG.get(language, {}).get("use_enhanced_extraction", False)

def get_language_patterns(language: str) -> dict:
    """Get language-specific patterns for entity extraction."""
    return LANGUAGE_CONFIG.get(language, {}).get("patterns", {})

def get_language_dictionaries(language: str) -> dict:
    """Get language-specific dictionaries for entity extraction."""
    return LANGUAGE_CONFIG.get(language, {}).get("dictionaries", {})

def get_language_relationship_patterns(language: str) -> list:
    """Get language-specific relationship patterns."""
    return LANGUAGE_CONFIG.get(language, {}).get("relationship_patterns", [])

def get_language_config(language: str) -> dict:
    """Get complete language configuration."""
    return LANGUAGE_CONFIG.get(language, {})

def get_min_entity_length(language: str) -> int:
    """Get minimum entity length for a language."""
    return LANGUAGE_CONFIG.get(language, {}).get("min_entity_length", 2)

def get_max_entity_length(language: str) -> int:
    """Get maximum entity length for a language."""
    return LANGUAGE_CONFIG.get(language, {}).get("max_entity_length", 50)

def get_confidence_threshold(language: str) -> float:
    """Get confidence threshold for a language."""
    return LANGUAGE_CONFIG.get(language, {}).get("confidence_threshold", 0.6)

def get_font_family(language: str) -> str:
    """Get font family for a language."""
    return LANGUAGE_CONFIG.get(language, {}).get("font_family", "Arial, sans-serif")

def get_encoding(language: str) -> str:
    """Get encoding for a language."""
    return LANGUAGE_CONFIG.get(language, {}).get("encoding", "utf-8")


# Removed duplicate functions that were using the old LANGUAGE_CONFIGS format


def detect_language_from_text(text: str) -> str:
    """
    Detect language from text using pattern matching.
    Returns the most likely language code.
    """
    if not text:
        return "en"
    
    # Count matches for each language
    language_scores = {}
    
    # Check for Russian characters
    russian_chars = len(re.findall(r'[А-ЯЁа-яё]', text))
    if russian_chars > 0:
        language_scores["ru"] = russian_chars
    
    # Check for Chinese characters
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    if chinese_chars > 0:
        language_scores["zh"] = chinese_chars
    
    # Check for English words
    english_words = len(re.findall(r'\b[A-Za-z]{2,}\b', text))
    if english_words > 0:
        language_scores["en"] = english_words
    
    # Find the language with the highest score
    if language_scores:
        best_language = max(language_scores.items(), key=lambda x: x[1])
        # Only return the language if it has a significant score
        if best_language[1] > 1:  # Threshold for confidence
            return best_language[0]
    
    # Default to English if no clear pattern is found
    return "en"


def detect_primary_language(text: str) -> str:
    """
    Detect the primary language from mixed-language text.
    This is especially useful for documents that contain multiple languages.
    """
    # For Russian specifically, look for Cyrillic characters
    russian_chars = len(re.findall(r'[А-ЯЁа-яё]', text))
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_words = len(re.findall(r'\b[A-Za-z]{2,}\b', text))
    
    # Calculate ratios
    total_chars = len(text.replace(' ', ''))
    if total_chars > 0:
        russian_ratio = russian_chars / total_chars
        chinese_ratio = chinese_chars / total_chars
        english_ratio = english_words / (len(text.split()) + 1)
        
        # Determine primary language based on character ratios
        if russian_ratio > 0.05:  # More than 5% Russian characters (lowered threshold)
            return "ru"
        elif chinese_ratio > 0.05:  # More than 5% Chinese characters (lowered threshold)
            return "zh"
        elif english_ratio > 0.3:  # More than 30% English words
            return "en"
    
    # Fallback to pattern-based detection
    return detect_language_from_text(text)
