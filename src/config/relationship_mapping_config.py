"""
Relationship Mapping Configuration
Configuration settings for multilingual relationship extraction with language-specific prompts.
"""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class RelationshipTypeConfig:
    """Configuration for relationship types."""
    name: str
    description: str
    examples: List[str]
    confidence_threshold: float = 0.7


@dataclass
class LanguageRelationshipConfig:
    """Configuration for language-specific relationship mapping."""
    language_code: str
    language_name: str
    relationship_types: Dict[str, RelationshipTypeConfig]
    prompts: Dict[str, str]
    patterns: Dict[str, List[str]]


# Comprehensive relationship types
RELATIONSHIP_TYPES = {
    "IS_A": RelationshipTypeConfig(
        name="IS_A",
        description="Taxonomic relationship indicating class membership",
        examples=["is a", "is an", "is a type of", "belongs to"],
        confidence_threshold=0.8
    ),
    "PART_OF": RelationshipTypeConfig(
        name="PART_OF",
        description="Meronymic relationship indicating component membership",
        examples=["part of", "component of", "member of", "belongs to"],
        confidence_threshold=0.8
    ),
    "LOCATED_IN": RelationshipTypeConfig(
        name="LOCATED_IN",
        description="Spatial relationship indicating location",
        examples=["located in", "situated in", "found in", "based in"],
        confidence_threshold=0.9
    ),
    "WORKS_FOR": RelationshipTypeConfig(
        name="WORKS_FOR",
        description="Employment or affiliation relationship",
        examples=["works for", "employed by", "affiliated with", "member of"],
        confidence_threshold=0.8
    ),
    "CREATED_BY": RelationshipTypeConfig(
        name="CREATED_BY",
        description="Creation or authorship relationship",
        examples=["created by", "developed by", "authored by", "founded by"],
        confidence_threshold=0.8
    ),
    "USES": RelationshipTypeConfig(
        name="USES",
        description="Usage or utilization relationship",
        examples=["uses", "utilizes", "employs", "applies"],
        confidence_threshold=0.7
    ),
    "IMPLEMENTS": RelationshipTypeConfig(
        name="IMPLEMENTS",
        description="Implementation or execution relationship",
        examples=["implements", "executes", "carries out", "performs"],
        confidence_threshold=0.8
    ),
    "SIMILAR_TO": RelationshipTypeConfig(
        name="SIMILAR_TO",
        description="Similarity or comparison relationship",
        examples=["similar to", "like", "comparable to", "resembles"],
        confidence_threshold=0.6
    ),
    "OPPOSES": RelationshipTypeConfig(
        name="OPPOSES",
        description="Opposition or conflict relationship",
        examples=["opposes", "against", "conflicts with", "contradicts"],
        confidence_threshold=0.8
    ),
    "SUPPORTS": RelationshipTypeConfig(
        name="SUPPORTS",
        description="Support or endorsement relationship",
        examples=["supports", "endorses", "backs", "promotes"],
        confidence_threshold=0.8
    ),
    "LEADS_TO": RelationshipTypeConfig(
        name="LEADS_TO",
        description="Causal or consequential relationship",
        examples=["leads to", "causes", "results in", "brings about"],
        confidence_threshold=0.7
    ),
    "DEPENDS_ON": RelationshipTypeConfig(
        name="DEPENDS_ON",
        description="Dependency or requirement relationship",
        examples=["depends on", "requires", "needs", "relies on"],
        confidence_threshold=0.8
    ),
    "RELATED_TO": RelationshipTypeConfig(
        name="RELATED_TO",
        description="General association or connection",
        examples=["related to", "connected to", "associated with", "linked to"],
        confidence_threshold=0.5
    )
}


# Language-specific configurations
LANGUAGE_RELATIONSHIP_CONFIGS = {
    "en": LanguageRelationshipConfig(
        language_code="en",
        language_name="English",
        relationship_types=RELATIONSHIP_TYPES,
        prompts={
            "main": """
You are an expert relationship extraction system. Analyze the relationships between entities in the given text.

Instructions:
1. Identify all meaningful relationships between the provided entities
2. For each relationship, provide:
   - source: The source entity name (exact match from entities list)
   - target: The target entity name (exact match from entities list)
   - relationship_type: One of [IS_A, PART_OF, LOCATED_IN, WORKS_FOR, CREATED_BY, USES, IMPLEMENTS, SIMILAR_TO, OPPOSES, SUPPORTS, LEADS_TO, DEPENDS_ON, RELATED_TO]
   - confidence: A score between 0.0 and 1.0 based on your certainty
   - description: A clear description of the relationship
3. Only include relationships that are explicitly mentioned or strongly implied in the text
4. Return ONLY valid JSON in the exact format specified
5. If no clear relationships exist, return an empty relationships array

Entities to analyze: {entities}

Text to analyze:
{text}

Expected JSON format:
{{
    "relationships": [
        {{
            "source": "entity_name",
            "target": "entity_name",
            "relationship_type": "relationship_type",
            "confidence": 0.95,
            "description": "clear description of the relationship"
        }}
    ]
}}

Important:
- Use exact entity names from the entities list
- Only create relationships that are clearly supported by the text
- If uncertain, use "RELATED_TO" as the relationship type
- Return only valid JSON, no additional text or explanations

Return only the JSON object, no additional text.
""",
            "fallback": "Create meaningful relationships between the entities based on their types and context."
        },
        patterns={
            "IS_A": ["is a", "is an", "is a type of", "belongs to", "category of"],
            "PART_OF": ["part of", "component of", "member of", "belongs to", "section of"],
            "LOCATED_IN": ["located in", "situated in", "found in", "based in", "in"],
            "WORKS_FOR": ["works for", "employed by", "affiliated with", "member of", "at"],
            "CREATED_BY": ["created by", "developed by", "authored by", "founded by", "by"],
            "USES": ["uses", "utilizes", "employs", "applies", "with"],
            "IMPLEMENTS": ["implements", "executes", "carries out", "performs", "does"],
            "SIMILAR_TO": ["similar to", "like", "comparable to", "resembles", "similar"],
            "OPPOSES": ["opposes", "against", "conflicts with", "contradicts", "versus"],
            "SUPPORTS": ["supports", "endorses", "backs", "promotes", "for"],
            "LEADS_TO": ["leads to", "causes", "results in", "brings about", "creates"],
            "DEPENDS_ON": ["depends on", "requires", "needs", "relies on", "needs"],
            "RELATED_TO": ["related to", "connected to", "associated with", "linked to", "and"]
        }
    ),
    "zh": LanguageRelationshipConfig(
        language_code="zh",
        language_name="Chinese",
        relationship_types=RELATIONSHIP_TYPES,
        prompts={
            "main": """
你是一个专业的关系提取系统。分析给定文本中实体之间的关系。

说明：
1. 识别提供的实体之间的所有有意义的关系
2. 对于每个关系，提供：
   - source: 源实体名称（与实体列表中的名称完全匹配）
   - target: 目标实体名称（与实体列表中的名称完全匹配）
   - relationship_type: 关系类型，从以下选择：[IS_A, PART_OF, LOCATED_IN, WORKS_FOR, CREATED_BY, USES, IMPLEMENTS, SIMILAR_TO, OPPOSES, SUPPORTS, LEADS_TO, DEPENDS_ON, RELATED_TO]
   - confidence: 基于你的确定性的0.0到1.0之间的分数
   - description: 关系的清晰描述
3. 只包含文本中明确提及或强烈暗示的关系
4. 仅返回指定格式的有效JSON
5. 如果没有明确的关系，返回空的关系数组

要分析的实体：{entities}

要分析的文本：
{text}

预期的JSON格式：
{{
    "relationships": [
        {{
            "source": "实体名称",
            "target": "实体名称",
            "relationship_type": "关系类型",
            "confidence": 0.95,
            "description": "关系的清晰描述"
        }}
    ]
}}

重要：
- 使用实体列表中的确切实体名称
- 只创建文本明确支持的关系
- 如果不确定，使用"RELATED_TO"作为关系类型
- 仅返回有效JSON，不要额外的文本或解释

仅返回JSON对象，不要额外的文本。
""",
            "fallback": "基于实体类型和上下文创建实体之间的有意义关系。"
        },
        patterns={
            "IS_A": ["是", "属于", "是一种", "是一个", "属于...类型"],
            "PART_OF": ["的一部分", "的组成部分", "的成员", "属于", "的部门"],
            "LOCATED_IN": ["位于", "坐落于", "在", "位于...中", "在...里"],
            "WORKS_FOR": ["为...工作", "受雇于", "隶属于", "是...的成员", "在...工作"],
            "CREATED_BY": ["由...创建", "由...开发", "由...撰写", "由...创立", "由...发明"],
            "USES": ["使用", "利用", "采用", "应用", "运用"],
            "IMPLEMENTS": ["实施", "执行", "执行", "执行", "实现"],
            "SIMILAR_TO": ["类似于", "像", "可比", "相似", "类似"],
            "OPPOSES": ["反对", "对抗", "与...冲突", "与...矛盾", "与...对立"],
            "SUPPORTS": ["支持", "赞同", "支持", "促进", "拥护"],
            "LEADS_TO": ["导致", "引起", "产生", "带来", "造成"],
            "DEPENDS_ON": ["依赖于", "需要", "需要", "依靠", "取决于"],
            "RELATED_TO": ["与...相关", "与...相连", "与...关联", "与...联系", "和"]
        }
    ),
    "ru": LanguageRelationshipConfig(
        language_code="ru",
        language_name="Russian",
        relationship_types=RELATIONSHIP_TYPES,
        prompts={
            "main": """
Вы являетесь экспертной системой извлечения отношений. Проанализируйте отношения между сущностями в данном тексте.

Инструкции:
1. Определите все значимые отношения между предоставленными сущностями
2. Для каждого отношения укажите:
   - source: Имя исходной сущности (точное совпадение из списка сущностей)
   - target: Имя целевой сущности (точное совпадение из списка сущностей)
   - relationship_type: Тип отношения из [IS_A, PART_OF, LOCATED_IN, WORKS_FOR, CREATED_BY, USES, IMPLEMENTS, SIMILAR_TO, OPPOSES, SUPPORTS, LEADS_TO, DEPENDS_ON, RELATED_TO]
   - confidence: Оценка от 0.0 до 1.0 на основе вашей уверенности
   - description: Четкое описание отношения
3. Включайте только отношения, которые явно упоминаются или сильно подразумеваются в тексте
4. Возвращайте ТОЛЬКО валидный JSON в указанном формате
5. Если четких отношений нет, верните пустой массив отношений

Сущности для анализа: {entities}

Текст для анализа:
{text}

Ожидаемый формат JSON:
{{
    "relationships": [
        {{
            "source": "имя_сущности",
            "target": "имя_сущности",
            "relationship_type": "тип_отношения",
            "confidence": 0.95,
            "description": "четкое описание отношения"
        }}
    ]
}}

Важно:
- Используйте точные имена сущностей из списка сущностей
- Создавайте только отношения, которые четко поддерживаются текстом
- Если не уверены, используйте "RELATED_TO" как тип отношения
- Возвращайте только валидный JSON, без дополнительного текста или объяснений

Возвращайте только JSON объект, без дополнительного текста.
""",
            "fallback": "Создайте значимые отношения между сущностями на основе их типов и контекста."
        },
        patterns={
            "IS_A": ["является", "есть", "является типом", "принадлежит к", "категория"],
            "PART_OF": ["часть", "компонент", "член", "принадлежит к", "секция"],
            "LOCATED_IN": ["расположен в", "находится в", "в", "базируется в", "внутри"],
            "WORKS_FOR": ["работает в", "нанят", "аффилирован с", "член", "в"],
            "CREATED_BY": ["создан", "разработан", "написан", "основан", "от"],
            "USES": ["использует", "применяет", "использует", "применяет", "с"],
            "IMPLEMENTS": ["реализует", "выполняет", "осуществляет", "выполняет", "делает"],
            "SIMILAR_TO": ["похож на", "как", "сравним с", "напоминает", "похож"],
            "OPPOSES": ["против", "против", "конфликтует с", "противоречит", "против"],
            "SUPPORTS": ["поддерживает", "одобряет", "поддерживает", "продвигает", "за"],
            "LEADS_TO": ["приводит к", "вызывает", "результат", "приносит", "создает"],
            "DEPENDS_ON": ["зависит от", "требует", "нуждается", "полагается на", "нуждается"],
            "RELATED_TO": ["связан с", "связан с", "связан с", "связан с", "и"]
        }
    )
}


def get_language_relationship_config(language_code: str) -> LanguageRelationshipConfig:
    """Get language-specific relationship configuration."""
    return LANGUAGE_RELATIONSHIP_CONFIGS.get(language_code.lower(), LANGUAGE_RELATIONSHIP_CONFIGS["en"])


def get_relationship_types(language_code: str) -> Dict[str, RelationshipTypeConfig]:
    """Get relationship types for a specific language."""
    config = get_language_relationship_config(language_code)
    return config.relationship_types


def get_relationship_prompts(language_code: str) -> Dict[str, str]:
    """Get relationship prompts for a specific language."""
    config = get_language_relationship_config(language_code)
    return config.prompts


def get_relationship_patterns(language_code: str) -> Dict[str, List[str]]:
    """Get relationship patterns for a specific language."""
    config = get_language_relationship_config(language_code)
    return config.patterns


def get_main_prompt(language_code: str, entities: List[str], text: str) -> str:
    """Get the main relationship mapping prompt for a specific language."""
    config = get_language_relationship_config(language_code)
    prompt_template = config.prompts["main"]
    return prompt_template.format(entities=entities, text=text)


def get_fallback_prompt(language_code: str) -> str:
    """Get the fallback relationship mapping prompt for a specific language."""
    config = get_language_relationship_config(language_code)
    return config.prompts["fallback"]
