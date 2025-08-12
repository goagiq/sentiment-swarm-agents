"""
Enhanced Chinese Entity Extraction for Knowledge Graph
Improved entity extraction specifically optimized for Chinese content.
"""

import re
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Entity:
    text: str
    entity_type: str
    confidence: float
    start_pos: int
    end_pos: int
    language: str = "zh"


class EnhancedChineseEntityExtractor:
    """Enhanced entity extractor specifically for Chinese content."""
    
    def __init__(self):
        # Enhanced Chinese entity patterns (Phase 6.2 improvements)
        self.chinese_patterns = {
            'PERSON': [
                # Improved patterns with word boundaries
                r'\b[\u4e00-\u9fff]{2,4}\b',  # 2-4 character names with boundaries
                r'\b[\u4e00-\u9fff]+\s*[先生|女士|教授|博士|院士|主席|总理|部长]\b',  # Titles with boundaries
                r'\b[\u4e00-\u9fff]{2,4}\s*[先生|女士]\b',  # Name + title combinations
            ],
            'ORGANIZATION': [
                # More specific organization patterns
                r'\b[\u4e00-\u9fff]+(?:公司|集团|企业|大学|学院|研究所|研究院|医院|银行|政府|部门)\b',
                r'\b[\u4e00-\u9fff]+(?:科技|技术|信息|网络|软件|硬件|生物|医药|金融|教育|文化)\b',
                r'\b[\u4e00-\u9fff]{2,6}(?:公司|集团|企业)\b',  # Specific company patterns
            ],
            'LOCATION': [
                # Improved location patterns
                r'\b[\u4e00-\u9fff]+(?:市|省|县|区|国|州|城|镇|村)\b',
                r'\b[\u4e00-\u9fff]+(?:山|河|湖|海|江|河|岛|湾)\b',
                r'\b[\u4e00-\u9fff]{2,4}(?:市|省|国)\b',  # Specific location patterns
            ],
            'CONCEPT': [
                # More comprehensive technical term patterns
                r'\b(?:人工智能|机器学习|深度学习|神经网络|自然语言处理|计算机视觉)\b',
                r'\b(?:量子计算|区块链|云计算|大数据|物联网|5G|6G)\b',
                r'\b(?:虚拟现实|增强现实|混合现实|元宇宙|数字化转型)\b',
                r'\b(?:数字经济|智能制造|绿色能源|可持续发展)\b',
            ]
        }
        
        # Enhanced Chinese prompts
        self.enhanced_prompts = {
            'comprehensive': """
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
""",
            
            'person_names': """
请从以下中文文本中提取所有人名：

文本：{text}

请识别以下类型的人名：
- 政治人物：习近平、李克强、王毅等
- 商业领袖：马云、马化腾、任正非等
- 学者专家：李国杰、潘建伟、张首晟等
- 其他知名人物

请按JSON格式返回：
{{
    "persons": ["姓名1", "姓名2", ...]
}}
""",
            
            'organizations': """
请从以下中文文本中提取所有组织名称：

文本：{text}

请识别以下类型的组织：
- 公司企业：华为、阿里巴巴、腾讯、百度等
- 教育机构：清华大学、北京大学、中科院等
- 政府部门：国务院、发改委、科技部等
- 研究机构：计算所、自动化所、软件所等

请按JSON格式返回：
{{
    "organizations": ["组织名1", "组织名2", ...]
}}
""",
            
            'technical_terms': """
请从以下中文文本中提取所有技术术语和概念：

文本：{text}

请识别以下类型的技术术语：
- AI技术：人工智能、机器学习、深度学习、神经网络等
- 新兴技术：5G、量子计算、区块链、云计算等
- 专业概念：自然语言处理、计算机视觉、大数据等

请按JSON格式返回：
{{
    "concepts": ["术语1", "术语2", ...]
}}
"""
        }
    
    async def extract_entities_enhanced(self, text: str, entity_type: str = "all") -> List[Entity]:
        """Enhanced entity extraction with multiple strategies."""
        entities = []
        
        # Strategy 1: Enhanced LLM-based extraction
        llm_entities = await self._extract_with_enhanced_prompt(text, entity_type)
        entities.extend(llm_entities)
        
        # Strategy 2: Pattern-based extraction
        pattern_entities = self._extract_with_patterns(text)
        entities.extend(pattern_entities)
        
        # Strategy 3: Dictionary-based extraction
        dict_entities = self._extract_with_dictionary(text)
        entities.extend(dict_entities)
        
        # Remove duplicates and merge
        return self._merge_and_clean_entities(entities)
    
    async def _extract_with_enhanced_prompt(self, text: str, entity_type: str) -> List[Entity]:
        """Extract entities using enhanced prompts."""
        if entity_type == "all":
            prompt = self.enhanced_prompts['comprehensive'].format(text=text)
        elif entity_type == "PERSON":
            prompt = self.enhanced_prompts['person_names'].format(text=text)
        elif entity_type == "ORGANIZATION":
            prompt = self.enhanced_prompts['organizations'].format(text=text)
        elif entity_type == "CONCEPT":
            prompt = self.enhanced_prompts['technical_terms'].format(text=text)
        else:
            prompt = self.enhanced_prompts['comprehensive'].format(text=text)
        
        # This would call the LLM - placeholder for now
        # response = await self.strands_agent.run(prompt)
        # return self._parse_llm_response(response)
        
        return []
    
    def _extract_with_patterns(self, text: str) -> List[Entity]:
        """Extract entities using regex patterns."""
        entities = []
        
        for entity_type, patterns in self.chinese_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    entity = Entity(
                        text=match.group(),
                        entity_type=entity_type,
                        confidence=0.8,  # Pattern-based confidence
                        start_pos=match.start(),
                        end_pos=match.end(),
                        language="zh"
                    )
                    entities.append(entity)
        
        return entities
    
    def _extract_with_dictionary(self, text: str) -> List[Entity]:
        """Extract entities using dictionary lookup."""
        # Chinese entity dictionaries
        dictionaries = {
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
        
        entities = []
        for entity_type, entity_list in dictionaries.items():
            for entity_text in entity_list:
                if entity_text in text:
                    start_pos = text.find(entity_text)
                    entity = Entity(
                        text=entity_text,
                        entity_type=entity_type,
                        confidence=0.9,  # Dictionary-based confidence
                        start_pos=start_pos,
                        end_pos=start_pos + len(entity_text),
                        language="zh"
                    )
                    entities.append(entity)
        
        return entities
    
    def _merge_and_clean_entities(self, entities: List[Entity]) -> List[Entity]:
        """Merge and clean extracted entities."""
        # Remove duplicates
        unique_entities = {}
        for entity in entities:
            key = (entity.text, entity.entity_type)
            if key not in unique_entities or entity.confidence > unique_entities[key].confidence:
                unique_entities[key] = entity
        
        # Sort by confidence
        cleaned_entities = list(unique_entities.values())
        cleaned_entities.sort(key=lambda x: x.confidence, reverse=True)
        
        return cleaned_entities
    
    def _parse_llm_response(self, response: str) -> List[Entity]:
        """Parse LLM response to extract entities."""
        entities = []
        
        try:
            # Try to parse as JSON
            if '{' in response and '}' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]
                
                data = json.loads(json_str)
                
                if 'entities' in data:
                    for entity_data in data['entities']:
                        entity = Entity(
                            text=entity_data.get('text', ''),
                            entity_type=entity_data.get('type', 'UNKNOWN'),
                            confidence=entity_data.get('confidence', 0.5),
                            start_pos=0,
                            end_pos=len(entity_data.get('text', '')),
                            language="zh"
                        )
                        entities.append(entity)
                
                elif 'persons' in data:
                    for person in data['persons']:
                        entity = Entity(
                            text=person,
                            entity_type='PERSON',
                            confidence=0.8,
                            start_pos=0,
                            end_pos=len(person),
                            language="zh"
                        )
                        entities.append(entity)
                
                elif 'organizations' in data:
                    for org in data['organizations']:
                        entity = Entity(
                            text=org,
                            entity_type='ORGANIZATION',
                            confidence=0.8,
                            start_pos=0,
                            end_pos=len(org),
                            language="zh"
                        )
                        entities.append(entity)
                
                elif 'concepts' in data:
                    for concept in data['concepts']:
                        entity = Entity(
                            text=concept,
                            entity_type='CONCEPT',
                            confidence=0.8,
                            start_pos=0,
                            end_pos=len(concept),
                            language="zh"
                        )
                        entities.append(entity)
        
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback: extract entities using patterns
            pass
        
        return entities


# Chinese entity validation rules
class ChineseEntityValidator:
    """Validate and clean Chinese entities."""
    
    @staticmethod
    def validate_person_name(name: str) -> bool:
        """Validate Chinese person names."""
        # Chinese names are typically 2-4 characters
        if len(name) < 2 or len(name) > 4:
            return False
        
        # Should contain only Chinese characters
        if not re.match(r'^[\u4e00-\u9fff]+$', name):
            return False
        
        # Common Chinese surnames
        common_surnames = [
            '王', '李', '张', '刘', '陈', '杨', '赵', '黄', '周', '吴',
            '徐', '孙', '胡', '朱', '高', '林', '何', '郭', '马', '罗',
            '梁', '宋', '郑', '谢', '韩', '唐', '冯', '于', '董', '萧'
        ]
        
        # Check if first character is a common surname
        if name[0] in common_surnames:
            return True
        
        return True  # Allow other names too
    
    @staticmethod
    def validate_organization_name(name: str) -> bool:
        """Validate Chinese organization names (Phase 6.2 improvements)."""
        # Should be at least 2 characters
        if len(name) < 2:
            return False
        
        # Should contain Chinese characters
        if not re.search(r'[\u4e00-\u9fff]', name):
            return False
        
        # Common organization suffixes (relaxed validation)
        org_suffixes = ['公司', '集团', '企业', '大学', '学院', '研究所', '研究院', '医院', '银行', '政府', '部门']
        
        # Check for suffixes
        if any(suffix in name for suffix in org_suffixes):
            return True
        
        # Check for common organization names without suffixes
        common_orgs = ['华为', '阿里巴巴', '腾讯', '百度', '京东', '美团', '清华大学', '北京大学', '中科院']
        if name in common_orgs:
            return True
        
        # Check for organization-like patterns
        if re.search(r'[\u4e00-\u9fff]{2,6}', name):
            return True
        
        return False
    
    @staticmethod
    def validate_location_name(name: str) -> bool:
        """Validate Chinese location names (Phase 6.2 improvements)."""
        # Should be at least 2 characters
        if len(name) < 2:
            return False
        
        # Should contain Chinese characters
        if not re.search(r'[\u4e00-\u9fff]', name):
            return False
        
        # Common location suffixes
        loc_suffixes = ['市', '省', '县', '区', '国', '州', '城', '镇', '村']
        
        # Check for suffixes
        if any(suffix in name for suffix in loc_suffixes):
            return True
        
        # Check for common location names without suffixes
        common_locations = ['北京', '上海', '深圳', '广州', '杭州', '南京', '中国', '美国', '日本', '韩国', '德国', '法国']
        if name in common_locations:
            return True
        
        # Check for location-like patterns
        if re.search(r'[\u4e00-\u9fff]{2,4}', name):
            return True
        
        return False
    
    @staticmethod
    def validate_technical_term(term: str) -> bool:
        """Validate Chinese technical terms."""
        # Should be at least 2 characters
        if len(term) < 2:
            return False
        
        # Should contain Chinese characters
        if not re.search(r'[\u4e00-\u9fff]', term):
            return False
        
        # Common technical term patterns
        tech_patterns = [
            r'人工智能', r'机器学习', r'深度学习', r'神经网络',
            r'自然语言处理', r'计算机视觉', r'量子计算', r'区块链',
            r'云计算', r'大数据', r'物联网', r'虚拟现实'
        ]
        
        return any(re.search(pattern, term) for pattern in tech_patterns)
