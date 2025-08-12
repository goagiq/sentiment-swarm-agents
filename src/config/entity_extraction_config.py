"""
Entity Extraction Configuration
Configuration settings for multilingual entity extraction with enhanced categorization.
"""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class EntityTypeConfig:
    """Configuration for entity types."""
    name: str
    description: str
    examples: List[str]
    patterns: List[str]
    confidence_threshold: float = 0.7


@dataclass
class LanguageConfig:
    """Configuration for language-specific entity extraction."""
    language_code: str
    language_name: str
    entity_types: Dict[str, EntityTypeConfig]
    common_entities: Dict[str, List[str]]
    patterns: Dict[str, List[str]]


# Comprehensive entity patterns for enhanced categorization
COMPREHENSIVE_PERSON_PATTERNS = [
    # Common names
    'trump', 'donald', 'biden', 'joe', 'obama', 'barack', 'clinton', 'hillary', 'bush', 'george',
    'whitmer', 'gretchen', 'pence', 'mike', 'harris', 'kamala', 'pelosi', 'nancy', 'mcconnell', 'mitch',
    'schumer', 'chuck', 'warren', 'elizabeth', 'sanders', 'bernie', 'aoc', 'ocasio', 'cortez',
    'musk', 'elon', 'bezos', 'jeff', 'gates', 'bill', 'zuckerberg', 'mark', 'cook', 'tim',
    'jobs', 'steve', 'page', 'larry', 'brin', 'sergey', 'ma', 'jack', 'pony', 'hua',
    
    # Titles and roles
    'president', 'governor', 'senator', 'congressman', 'congresswoman', 'mayor', 'secretary',
    'minister', 'prime', 'chancellor', 'director', 'chief', 'executive', 'officer', 'ceo',
    'professor', 'doctor', 'dr', 'mr', 'mrs', 'ms', 'miss', 'sir', 'madam', 'lady',
    'captain', 'general', 'colonel', 'lieutenant', 'sergeant', 'officer', 'detective',
    'judge', 'lawyer', 'attorney', 'counsel', 'advocate', 'solicitor', 'barrister',
    
    # Academic and professional titles
    'professor', 'associate', 'assistant', 'lecturer', 'instructor', 'researcher', 'scientist',
    'engineer', 'architect', 'designer', 'consultant', 'analyst', 'specialist', 'expert',
    'author', 'writer', 'journalist', 'reporter', 'editor', 'publisher', 'producer',
    'artist', 'musician', 'actor', 'actress', 'director', 'filmmaker', 'photographer'
]

COMPREHENSIVE_LOCATION_PATTERNS = [
    # Countries
    'michigan', 'california', 'texas', 'florida', 'new york', 'washington', 'usa', 'america',
    'china', 'chinese', 'mexico', 'canada', 'britain', 'france', 'germany', 'italy', 'spain',
    'japan', 'japanese', 'korea', 'korean', 'india', 'russia', 'russian', 'brazil', 'australia',
    'england', 'scotland', 'wales', 'ireland', 'sweden', 'norway', 'denmark', 'finland',
    'netherlands', 'belgium', 'switzerland', 'austria', 'poland', 'czech', 'hungary', 'romania',
    'bulgaria', 'greece', 'turkey', 'israel', 'egypt', 'south africa', 'nigeria', 'kenya',
    'argentina', 'chile', 'peru', 'colombia', 'venezuela', 'ecuador', 'bolivia', 'paraguay',
    
    # States and provinces
    'alabama', 'alaska', 'arizona', 'arkansas', 'colorado', 'connecticut', 'delaware',
    'georgia', 'hawaii', 'idaho', 'illinois', 'indiana', 'iowa', 'kansas', 'kentucky',
    'louisiana', 'maine', 'maryland', 'massachusetts', 'minnesota', 'mississippi', 'missouri',
    'montana', 'nebraska', 'nevada', 'new hampshire', 'new jersey', 'new mexico', 'north carolina',
    'north dakota', 'ohio', 'oklahoma', 'oregon', 'pennsylvania', 'rhode island', 'south carolina',
    'south dakota', 'tennessee', 'utah', 'vermont', 'virginia', 'west virginia', 'wisconsin', 'wyoming',
    
    # Cities
    'new york', 'los angeles', 'chicago', 'houston', 'phoenix', 'philadelphia', 'san antonio',
    'san diego', 'dallas', 'san jose', 'austin', 'jacksonville', 'fort worth', 'columbus',
    'charlotte', 'san francisco', 'indianapolis', 'seattle', 'denver', 'washington', 'boston',
    'nashville', 'baltimore', 'portland', 'las vegas', 'milwaukee', 'albuquerque', 'tucson',
    'fresno', 'sacramento', 'atlanta', 'long beach', 'colorado springs', 'raleigh', 'miami',
    'virginia beach', 'oakland', 'minneapolis', 'tampa', 'tulsa', 'arlington', 'new orleans',
    
    # Geographic features
    'mountain', 'river', 'lake', 'ocean', 'sea', 'bay', 'gulf', 'strait', 'canal', 'bridge',
    'valley', 'canyon', 'desert', 'forest', 'jungle', 'island', 'peninsula', 'cape', 'cliff',
    'volcano', 'glacier', 'waterfall', 'spring', 'creek', 'stream', 'pond', 'marsh', 'swamp'
]

COMPREHENSIVE_ORGANIZATION_PATTERNS = [
    # Government bodies
    'government', 'administration', 'congress', 'senate', 'house', 'parliament', 'assembly',
    'council', 'committee', 'department', 'ministry', 'agency', 'bureau', 'office', 'commission',
    'federal', 'state', 'local', 'municipal', 'county', 'city', 'town', 'village',
    'executive', 'legislative', 'judicial', 'supreme court', 'district court', 'appeals court',
    
    # Companies and corporations
    'microsoft', 'apple', 'google', 'amazon', 'facebook', 'meta', 'netflix', 'tesla', 'spacex',
    'twitter', 'linkedin', 'uber', 'lyft', 'airbnb', 'spotify', 'salesforce', 'oracle', 'ibm',
    'intel', 'amd', 'nvidia', 'cisco', 'dell', 'hp', 'lenovo', 'samsung', 'sony', 'nintendo',
    'disney', 'warner', 'paramount', 'universal', 'fox', 'cnn', 'bbc', 'nbc', 'abc', 'cbs',
    
    # Educational institutions
    'university', 'college', 'school', 'institute', 'academy', 'conservatory', 'seminary',
    'harvard', 'stanford', 'mit', 'yale', 'princeton', 'columbia', 'penn', 'cornell', 'brown',
    'dartmouth', 'berkeley', 'ucla', 'usc', 'nyu', 'georgetown', 'duke', 'northwestern',
    
    # International organizations
    'united nations', 'who', 'unicef', 'unesco', 'world bank', 'imf', 'wto', 'nato', 'eu',
    'asean', 'african union', 'oas', 'g7', 'g20', 'brics', 'opec', 'wto', 'red cross',
    
    # Financial institutions
    'bank', 'credit union', 'savings', 'investment', 'insurance', 'mutual fund', 'hedge fund',
    'jpmorgan', 'bank of america', 'wells fargo', 'citigroup', 'goldman sachs', 'morgan stanley',
    'federal reserve', 'treasury', 'irs', 'sec', 'fdic', 'fannie mae', 'freddie mac'
]

COMPREHENSIVE_CONCEPT_PATTERNS = [
    # Abstract ideas and policies
    'tariffs', 'policy', 'trade', 'economics', 'finance', 'banking', 'investment', 'taxation',
    'regulation', 'legislation', 'law', 'justice', 'democracy', 'republic', 'freedom', 'liberty',
    'equality', 'justice', 'rights', 'responsibility', 'duty', 'obligation', 'privilege',
    'authority', 'power', 'influence', 'control', 'management', 'leadership', 'governance',
    
    # Technology and innovation
    'artificial intelligence', 'machine learning', 'deep learning', 'neural networks', 'blockchain',
    'cryptocurrency', 'bitcoin', 'ethereum', 'cloud computing', 'big data', 'analytics',
    'cybersecurity', 'privacy', 'encryption', 'authentication', 'authorization', 'biometrics',
    'robotics', 'automation', 'internet of things', 'iot', 'virtual reality', 'augmented reality',
    
    # Social and cultural concepts
    'culture', 'society', 'community', 'family', 'education', 'healthcare', 'medicine', 'science',
    'research', 'development', 'innovation', 'creativity', 'art', 'music', 'literature', 'philosophy',
    'religion', 'spirituality', 'ethics', 'morality', 'values', 'beliefs', 'traditions', 'customs',
    
    # Business and economic concepts
    'business', 'commerce', 'industry', 'manufacturing', 'production', 'distribution', 'marketing',
    'advertising', 'branding', 'customer service', 'quality', 'efficiency', 'productivity', 'profit',
    'revenue', 'cost', 'expense', 'budget', 'financial planning', 'risk management', 'strategy'
]

COMPREHENSIVE_OBJECT_PATTERNS = [
    # Physical objects and products
    'computer', 'phone', 'smartphone', 'tablet', 'laptop', 'desktop', 'server', 'router', 'modem',
    'car', 'truck', 'bus', 'train', 'plane', 'airplane', 'helicopter', 'ship', 'boat', 'submarine',
    'building', 'house', 'apartment', 'office', 'factory', 'warehouse', 'store', 'shop', 'mall',
    'furniture', 'table', 'chair', 'bed', 'sofa', 'desk', 'cabinet', 'shelf', 'mirror', 'lamp',
    
    # Technology products
    'software', 'application', 'app', 'program', 'system', 'platform', 'database', 'server',
    'hardware', 'processor', 'memory', 'storage', 'display', 'screen', 'monitor', 'keyboard',
    'mouse', 'printer', 'scanner', 'camera', 'microphone', 'speaker', 'headphone', 'earphone',
    
    # Materials and substances
    'steel', 'aluminum', 'copper', 'iron', 'gold', 'silver', 'platinum', 'diamond', 'ruby',
    'plastic', 'glass', 'ceramic', 'wood', 'stone', 'concrete', 'cement', 'asphalt', 'fabric',
    'paper', 'cardboard', 'leather', 'rubber', 'synthetic', 'natural', 'organic', 'inorganic'
]

COMPREHENSIVE_PROCESS_PATTERNS = [
    # Activities and operations
    'manufacturing', 'production', 'assembly', 'fabrication', 'construction', 'building',
    'development', 'design', 'engineering', 'testing', 'quality control', 'inspection',
    'maintenance', 'repair', 'service', 'installation', 'configuration', 'deployment',
    
    # Business processes
    'planning', 'strategizing', 'analysis', 'research', 'investigation', 'evaluation',
    'assessment', 'review', 'audit', 'compliance', 'certification', 'accreditation',
    'training', 'education', 'learning', 'teaching', 'coaching', 'mentoring', 'consulting',
    
    # Technical processes
    'programming', 'coding', 'debugging', 'testing', 'deployment', 'integration', 'migration',
    'backup', 'recovery', 'optimization', 'tuning', 'scaling', 'monitoring', 'logging',
    'authentication', 'authorization', 'encryption', 'decryption', 'compression', 'decompression',
    
    # Scientific processes
    'experimentation', 'observation', 'measurement', 'calculation', 'computation', 'simulation',
    'modeling', 'prediction', 'forecasting', 'estimation', 'approximation', 'interpolation',
    'extrapolation', 'regression', 'classification', 'clustering', 'optimization', 'minimization'
]

# Generic entity type configurations with comprehensive patterns
GENERIC_ENTITY_TYPES = {
    "PERSON": EntityTypeConfig(
        name="PERSON",
        description="Individual people, politicians, leaders, public figures",
        examples=["John Doe", "President", "CEO", "Professor"],
        patterns=COMPREHENSIVE_PERSON_PATTERNS,
        confidence_threshold=0.8
    ),
    "ORGANIZATION": EntityTypeConfig(
        name="ORGANIZATION", 
        description="Companies, governments, institutions, agencies, groups",
        examples=["Microsoft", "Government", "University", "Institute"],
        patterns=COMPREHENSIVE_ORGANIZATION_PATTERNS,
        confidence_threshold=0.8
    ),
    "LOCATION": EntityTypeConfig(
        name="LOCATION",
        description="Countries, states, cities, regions, places", 
        examples=["New York", "United States", "Europe", "Asia"],
        patterns=COMPREHENSIVE_LOCATION_PATTERNS,
        confidence_threshold=0.9
    ),
    "CONCEPT": EntityTypeConfig(
        name="CONCEPT",
        description="Abstract ideas, policies, theories, technical terms",
        examples=["Artificial Intelligence", "Machine Learning", "Policy", "Theory"],
        patterns=COMPREHENSIVE_CONCEPT_PATTERNS,
        confidence_threshold=0.7
    ),
    "OBJECT": EntityTypeConfig(
        name="OBJECT",
        description="Physical objects, products, materials, devices",
        examples=["Computer", "Car", "Building", "Phone"],
        patterns=COMPREHENSIVE_OBJECT_PATTERNS,
        confidence_threshold=0.7
    ),
    "PROCESS": EntityTypeConfig(
        name="PROCESS",
        description="Activities, operations, procedures, methods",
        examples=["Manufacturing", "Development", "Training", "Analysis"],
        patterns=COMPREHENSIVE_PROCESS_PATTERNS,
        confidence_threshold=0.7
    )
}


# Language-specific configurations
LANGUAGE_CONFIGS = {
    "en": LanguageConfig(
        language_code="en",
        language_name="English",
        entity_types=GENERIC_ENTITY_TYPES,
        common_entities={
            "PERSON": ["Donald Trump", "Joe Biden", "Barack Obama", "Elon Musk"],
            "ORGANIZATION": ["Microsoft", "Apple", "Google", "Amazon", "US Government"],
            "LOCATION": ["United States", "China", "New York", "California"],
            "CONCEPT": ["Artificial Intelligence", "Machine Learning", "Blockchain", "Cloud Computing"]
        },
        patterns={
            "PERSON": [r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'],
            "ORGANIZATION": [r'\b[A-Z][a-z]+ (Corp|Inc|Ltd|LLC|University|Institute)\b'],
            "LOCATION": [r'\b[A-Z][a-z]+ (City|State|Country|Province)\b'],
            "CONCEPT": [r'\b[A-Z][a-z]+ [A-Z][a-z]+\b']
        }
    ),
    "zh": LanguageConfig(
        language_code="zh",
        language_name="Chinese",
        entity_types={
            "PERSON": EntityTypeConfig(
                name="PERSON",
                description="Individual people, politicians, leaders, public figures",
                examples=["习近平", "马云", "李彦宏", "主席", "总理"],
                patterns=[r'[\u4e00-\u9fff]{2,4}', r'[\u4e00-\u9fff]+\s*[先生|女士|教授|博士|院士|主席|总理|部长]'],
                confidence_threshold=0.8
            ),
            "ORGANIZATION": EntityTypeConfig(
                name="ORGANIZATION",
                description="Companies, governments, institutions, agencies, groups", 
                examples=["华为", "阿里巴巴", "清华大学", "中科院", "政府"],
                patterns=[r'[\u4e00-\u9fff]+(?:公司|集团|企业|大学|学院|研究所|研究院|医院|银行|政府|部门)'],
                confidence_threshold=0.8
            ),
            "LOCATION": EntityTypeConfig(
                name="LOCATION",
                description="Countries, states, cities, regions, places",
                examples=["北京", "上海", "中国", "美国", "深圳"],
                patterns=[r'[\u4e00-\u9fff]+(?:市|省|县|区|国|州|城|镇|村)'],
                confidence_threshold=0.9
            ),
            "CONCEPT": EntityTypeConfig(
                name="CONCEPT", 
                description="Abstract ideas, policies, theories, technical terms",
                examples=["人工智能", "机器学习", "区块链", "云计算"],
                patterns=[r'人工智能|机器学习|深度学习|神经网络|自然语言处理|计算机视觉'],
                confidence_threshold=0.7
            )
        },
        common_entities={
            "PERSON": ["习近平", "李克强", "马云", "马化腾", "任正非", "李彦宏"],
            "ORGANIZATION": ["华为", "阿里巴巴", "腾讯", "百度", "清华大学", "北京大学", "中科院"],
            "LOCATION": ["北京", "上海", "深圳", "广州", "中国", "美国", "日本"],
            "CONCEPT": ["人工智能", "机器学习", "深度学习", "神经网络", "量子计算", "区块链"]
        },
        patterns={
            "PERSON": [r'[\u4e00-\u9fff]{2,4}', r'[\u4e00-\u9fff]+\s*[先生|女士|教授|博士|院士|主席|总理|部长]'],
            "ORGANIZATION": [r'[\u4e00-\u9fff]+(?:公司|集团|企业|大学|学院|研究所|研究院|医院|银行|政府|部门)'],
            "LOCATION": [r'[\u4e00-\u9fff]+(?:市|省|县|区|国|州|城|镇|村)'],
            "CONCEPT": [r'人工智能|机器学习|深度学习|神经网络|自然语言处理|计算机视觉']
        }
    ),
    "ru": LanguageConfig(
        language_code="ru",
        language_name="Russian",
        entity_types={
            "PERSON": EntityTypeConfig(
                name="PERSON",
                description="Individual people, politicians, leaders, public figures",
                examples=["Владимир Путин", "Дмитрий Медведев", "Алексей Навальный", "Президент", "Премьер"],
                patterns=[r'[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?', r'[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.'],
                confidence_threshold=0.8
            ),
            "ORGANIZATION": EntityTypeConfig(
                name="ORGANIZATION",
                description="Companies, governments, institutions, agencies, groups",
                examples=["Газпром", "Сбербанк", "Российская Федерация", "МГУ", "РАН"],
                patterns=[r'[А-ЯЁ][а-яё]+(?:\s+[а-яё]+)*(?:ООО|ОАО|ЗАО|ПАО|ГК|Корпорация|Компания)', r'[А-ЯЁ][а-яё]+(?:\s+[а-яё]+)*(?:Университет|Институт|Академия|Университет)'],
                confidence_threshold=0.8
            ),
            "LOCATION": EntityTypeConfig(
                name="LOCATION",
                description="Countries, states, cities, regions, places",
                examples=["Москва", "Санкт-Петербург", "Россия", "Европа", "Азия"],
                patterns=[r'[А-ЯЁ][а-яё]+(?:\s+[а-яё]+)*(?:город|область|край|республика|район)', r'[А-ЯЁ][а-яё]+(?:\s+[а-яё]+)*(?:улица|проспект|переулок|площадь)'],
                confidence_threshold=0.9
            ),
            "CONCEPT": EntityTypeConfig(
                name="CONCEPT",
                description="Abstract ideas, policies, theories, technical terms",
                examples=["Искусственный интеллект", "Машинное обучение", "Блокчейн", "Облачные вычисления"],
                patterns=[r'искусственный интеллект|машинное обучение|глубокое обучение', r'блокчейн|облачные вычисления|большие данные|интернет вещей'],
                confidence_threshold=0.7
            )
        },
        common_entities={
            "PERSON": ["Владимир Путин", "Дмитрий Медведев", "Алексей Навальный", "Сергей Лавров", "Антон Силуанов", "Эльвира Набиуллина"],
            "ORGANIZATION": ["Газпром", "Сбербанк", "Роснефть", "Лукойл", "МГУ", "РАН", "Правительство России"],
            "LOCATION": ["Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург", "Россия", "США", "Китай"],
            "CONCEPT": ["Искусственный интеллект", "Машинное обучение", "Глубокое обучение", "Нейронные сети", "Квантовые вычисления", "Блокчейн"]
        },
        patterns={
            "PERSON": [r'[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?', r'[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.'],
            "ORGANIZATION": [r'[А-ЯЁ][а-яё]+(?:\s+[а-яё]+)*(?:ООО|ОАО|ЗАО|ПАО|ГК|Корпорация|Компания)', r'[А-ЯЁ][а-яё]+(?:\s+[а-яё]+)*(?:Университет|Институт|Академия|Университет)'],
            "LOCATION": [r'[А-ЯЁ][а-яё]+(?:\s+[а-яё]+)*(?:город|область|край|республика|район)', r'[А-ЯЁ][а-яё]+(?:\s+[а-яё]+)*(?:улица|проспект|переулок|площадь)'],
            "CONCEPT": [r'искусственный интеллект|машинное обучение|глубокое обучение', r'блокчейн|облачные вычисления|большие данные|интернет вещей']
        }
    )
}


def get_language_config(language_code: str) -> LanguageConfig:
    """Get language-specific configuration."""
    return LANGUAGE_CONFIGS.get(language_code, LANGUAGE_CONFIGS["en"])


def get_entity_types(language_code: str) -> Dict[str, EntityTypeConfig]:
    """Get entity types for a specific language."""
    config = get_language_config(language_code)
    return config.entity_types


def get_common_entities(language_code: str) -> Dict[str, List[str]]:
    """Get common entities for a specific language."""
    config = get_language_config(language_code)
    return config.common_entities


def get_patterns(language_code: str) -> Dict[str, List[str]]:
    """Get patterns for a specific language."""
    config = get_language_config(language_code)
    return config.patterns
