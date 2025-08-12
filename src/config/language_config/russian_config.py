"""
Russian language configuration for enhanced processing.
Enhanced with comprehensive regex patterns and grammar optimization.
"""

from typing import Dict, List
from .base_config import BaseLanguageConfig, EntityPatterns, ProcessingSettings


class RussianConfig(BaseLanguageConfig):
    """Russian language configuration with enhanced regex patterns and grammar support."""
    
    def __init__(self):
        super().__init__()
        self.language_code = "ru"
        self.language_name = "Russian"
        self.entity_patterns = self.get_entity_patterns()
        self.processing_settings = self.get_processing_settings()
        self.relationship_templates = self.get_relationship_templates()
        self.detection_patterns = self.get_detection_patterns()
        self.grammar_patterns = self.get_grammar_patterns()
        self.advanced_patterns = self.get_advanced_patterns()
    
    def get_entity_patterns(self) -> EntityPatterns:
        """Get enhanced Russian entity patterns."""
        return EntityPatterns(
            person=[
                # Basic Russian names
                r'\b[А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,})?\b',
                r'\b[А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.\b',
                # Names with titles
                r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,})*\s+(?:господин|госпожа|доктор|профессор)\b',
                # Patronymics
                r'\b[А-ЯЁ][а-яё]+(?:ович|евич|овна|евна)\b',
                # Academic titles
                r'\b(?:доктор|профессор|академик|кандидат)\s+[А-ЯЁ][а-яё]+\b',
                # Government titles
                r'\b(?:президент|министр|губернатор|мэр)\s+[А-ЯЁ][а-яё]+\b',
            ],
            organization=[
                # Business organizations
                r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[а-яё]{2,})*(?:ООО|ОАО|ЗАО|ПАО|ГК|Корпорация|Компания)\b',
                # Educational institutions
                r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[а-яё]{2,})*(?:Университет|Институт|Академия|Университет)\b',
                # Government organizations
                r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[а-яё]{2,})*(?:Правительство|Министерство|Агентство)\b',
                # Research institutions
                r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[а-яё]{2,})*(?:НИИ|Институт|Лаборатория|Центр)\b',
                # Media organizations
                r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[а-яё]{2,})*(?:Телеканал|Радио|Газета|Журнал)\b',
            ],
            location=[
                # Administrative divisions
                r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[а-яё]{2,})*(?:город|область|край|республика|район)\b',
                # Streets and addresses
                r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[а-яё]{2,})*(?:улица|проспект|переулок|площадь)\b',
                # Major cities
                r'\b(?:Москва|Санкт-Петербург|Новосибирск|Екатеринбург|Казань|Россия)\b',
                # Geographic features
                r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[а-яё]{2,})*(?:река|озеро|гора|море|остров)\b',
                # Historical places
                r'\b(?:Кремль|Красная площадь|Эрмитаж|Третьяковская галерея)\b',
            ],
            concept=[
                # Technology concepts
                r'\b(?:искусственный интеллект|машинное обучение|глубокое обучение)\b',
                r'\b(?:блокчейн|облачные вычисления|большие данные|интернет вещей)\b',
                r'\b(?:цифровая экономика|умное производство|зеленая энергия)\b',
                # Scientific concepts
                r'\b(?:квантовая физика|молекулярная биология|космические исследования)\b',
                # Economic concepts
                r'\b(?:рыночная экономика|глобализация|устойчивое развитие)\b',
                # Cultural concepts
                r'\b(?:русская литература|классическая музыка|изобразительное искусство)\b',
            ]
        )
    
    def get_grammar_patterns(self) -> Dict[str, List[str]]:
        """Get comprehensive grammar patterns for Russian."""
        return {
            "cases": [
                # Nominative case patterns
                r'\b[А-ЯЁ][а-яё]+(?:ый|ой|ий|ая|ое|ые|ие)\b',  # Adjectives
                r'\b[а-яё]+(?:ость|ость|ость|ость)\b',  # Abstract nouns
            ],
            "verb_forms": [
                # Verb conjugations
                r'\b[а-яё]+(?:ть|ться|л|ла|ло|ли|ет|ют|ит|ат)\b',
                r'\b[а-яё]+(?:ющий|ющий|ющий|ющий|ющий)\b',  # Participles
                r'\b[а-яё]+(?:емый|емый|емый|емый|емый)\b',  # Passive participles
            ],
            "prepositions": [
                # Common prepositions
                r'\b(?:в|на|с|по|для|от|до|из|за|под|над|между|через|около|вокруг)\b',
                r'\b(?:благодаря|вопреки|согласно|вследствие|ввиду|несмотря на)\b',
            ],
            "conjunctions": [
                # Coordinating conjunctions
                r'\b(?:и|а|но|или|либо|ни|да|зато|однако)\b',
                # Subordinating conjunctions
                r'\b(?:что|чтобы|если|когда|где|куда|откуда|почему|зачем|как)\b',
            ],
            "pronouns": [
                # Personal pronouns
                r'\b(?:я|ты|он|она|оно|мы|вы|они)\b',
                # Demonstrative pronouns
                r'\b(?:этот|тот|такой|таков|столько|это|то)\b',
                # Interrogative pronouns
                r'\b(?:кто|что|какой|какая|какое|какие|чей|чья|чьё|чьи)\b',
            ]
        }
    
    def get_advanced_patterns(self) -> Dict[str, List[str]]:
        """Get advanced Russian language patterns."""
        return {
            "scientific_terms": [
                r'\b(?:гипотеза|теория|эксперимент|исследование|анализ)\b',
                r'\b(?:методология|метод|подход|система|структура)\b',
                r'\b(?:результат|вывод|заключение|рекомендация|предложение)\b',
            ],
            "technical_terms": [
                r'\b(?:алгоритм|программа|система|база данных|интерфейс)\b',
                r'\b(?:протокол|стандарт|формат|код|функция)\b',
                r'\b(?:процесс|операция|команда|параметр|настройка)\b',
            ],
            "business_terms": [
                r'\b(?:стратегия|план|проект|бюджет|инвестиция)\b',
                r'\b(?:прибыль|доход|расход|стоимость|цена)\b',
                r'\b(?:рынок|конкуренция|клиент|партнер|поставщик)\b',
            ],
            "time_expressions": [
                r'\b(?:сегодня|вчера|завтра|сейчас|потом|раньше|позже)\b',
                r'\b(?:год|месяц|неделя|день|час|минута|секунда)\b',
                r'\b(?:январь|февраль|март|апрель|май|июнь|июль|август|сентябрь|октябрь|ноябрь|декабрь)\b',
            ],
            "measurement_units": [
                r'\b(?:метр|километр|сантиметр|миллиметр)\b',
                r'\b(?:килограмм|грамм|тонна|литр|миллилитр)\b',
                r'\b(?:рубль|доллар|евро|юань|иена)\b',
            ]
        }
    
    def get_processing_settings(self) -> ProcessingSettings:
        """Get enhanced Russian processing settings."""
        return ProcessingSettings(
            min_entity_length=3,  # Russian needs longer entities
            max_entity_length=50,
            confidence_threshold=0.7,
            use_enhanced_extraction=True,
            relationship_prompt_simplified=True,  # Russian works with simplified
            use_hierarchical_relationships=True,  # Enable for better organization
            entity_clustering_enabled=True,  # Enable for better grouping
            fallback_strategies=[
                "grammar_patterns",
                "advanced_patterns",
                "hierarchical",
                "semantic",
                "template"
            ]
        )
    
    def get_relationship_templates(self) -> Dict[str, str]:
        """Get enhanced Russian relationship templates."""
        return {
            "person_organization": "WORKS_FOR",
            "person_location": "LOCATED_IN",
            "organization_location": "LOCATED_IN",
            "concept_concept": "RELATED_TO",
            "person_person": "RELATED_TO",
            "organization_organization": "COLLABORATES_WITH",
            "concept_organization": "IMPLEMENTED_BY",
            "location_location": "NEAR_TO",
            "person_concept": "EXPERT_IN",
            "organization_concept": "SPECIALIZES_IN",
            # Russian-specific relationships
            "person_title": "HAS_TITLE",
            "organization_type": "IS_TYPE_OF",
            "location_administrative": "ADMINISTERS",
            "concept_field": "BELONGS_TO_FIELD",
        }
    
    def get_detection_patterns(self) -> List[str]:
        """Get enhanced Russian language detection patterns."""
        return [
            r'[а-яё]',  # Cyrillic characters
            r'\b(?:и|в|на|с|по|для|от|до|из|за|под|над|между|через)\b',  # Common prepositions
            r'\b(?:это|то|что|как|где|когда|почему|какой|какая|какое)\b',  # Common words
            r'\b(?:привет|мир|текст|документ|информация|данные)\b',  # Common nouns
            # Enhanced detection patterns
            r'\b(?:русский|российский|россия|москва|петербург)\b',  # Russian-specific terms
            r'\b(?:да|нет|хорошо|плохо|большой|маленький)\b',  # Common adjectives
            r'\b(?:говорить|делать|работать|жить|быть|стать)\b',  # Common verbs
        ]
    
    def get_hierarchical_relationship_prompt(self, entities: List[str], text: str) -> str:
        """Get Russian-specific hierarchical relationship prompt."""
        entities_str = ", ".join(entities[:15])
        return f"""
Проанализируйте следующий русский текст и определите иерархические отношения между этими сущностями: {entities_str}

Текст: {text[:1500]}...

Пожалуйста, определите следующие типы иерархических отношений:
1. Отношения подчинения: начальник-подчиненный, включение-включение
2. Парные отношения: сущности одного уровня
3. Пространственные отношения: географическое расположение, расположение учреждений
4. Функциональные отношения: рабочие отношения, профессиональные области
5. Концептуальные отношения: связанные концепции, технические связи
6. Организационные отношения: структура организаций, административные связи

Пожалуйста, предоставьте отношения в следующем формате:
Сущность1 | Тип отношения | Сущность2

Сосредоточьтесь на наиболее важных отношениях, обеспечивая точность и релевантность.
"""
    
    def get_entity_clustering_prompt(self, entities: List[str], text: str) -> str:
        """Get Russian-specific entity clustering prompt."""
        entities_str = ", ".join(entities[:20])
        return f"""
Сгруппируйте следующие русские сущности по семантическому сходству: {entities_str}

Текст: {text[:1000]}...

Пожалуйста, сгруппируйте по следующим критериям:
1. Группа лиц: связанные персоны
2. Группа организаций: связанные организации
3. Группа мест: связанные географические местоположения
4. Группа концепций: связанные технические концепции
5. Группа терминов: научные, технические, деловые термины

Создайте отношения для каждой группы, формат:
Сущность1 | Отношение в группе | Сущность2

Обеспечьте разумную группировку и значимые отношения.
"""
    
    def is_formal_russian(self, text: str) -> bool:
        """Detect if text contains formal Russian patterns."""
        import re
        formal_indicators = [
            r'уважаемый|господин|госпожа|прошу|сообщаю|довожу до вашего сведения',
            r'согласно|в соответствии с|на основании|в связи с|в целях',
            r'прошу рассмотреть|прошу предоставить|прошу направить',
            r'с уважением|искренне ваш|с наилучшими пожеланиями'
        ]
        
        for pattern in formal_indicators:
            if re.search(pattern, text.lower()):
                return True
        return False
    
    def get_formal_processing_settings(self) -> ProcessingSettings:
        """Get specialized processing settings for formal Russian."""
        return ProcessingSettings(
            min_entity_length=4,  # Formal Russian uses longer entities
            max_entity_length=60,  # Formal documents have longer entities
            confidence_threshold=0.8,  # Higher threshold for formal documents
            use_enhanced_extraction=True,
            relationship_prompt_simplified=False,  # Use detailed prompts for formal Russian
            use_hierarchical_relationships=True,
            entity_clustering_enabled=True,
            fallback_strategies=[
                "formal_patterns",
                "grammar_patterns",
                "hierarchical",
                "semantic"
            ]
        )
