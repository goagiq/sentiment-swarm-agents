"""
Chinese language configuration for enhanced processing.
Implements Phase 1 and Phase 2 improvements to address orphan nodes issue.
Enhanced with Classical Chinese patterns and comprehensive regex optimization.
"""

from typing import Dict, List
from .base_config import BaseLanguageConfig, EntityPatterns, ProcessingSettings


class ChineseConfig(BaseLanguageConfig):
    """Chinese language configuration with enhanced processing for orphan nodes and Classical Chinese support."""
    
    def __init__(self):
        super().__init__()
        self.language_code = "zh"
        self.language_name = "Chinese"
        self.entity_patterns = self.get_entity_patterns()
        self.processing_settings = self.get_processing_settings()
        self.relationship_templates = self.get_relationship_templates()
        self.detection_patterns = self.get_detection_patterns()
        self.classical_patterns = self.get_classical_chinese_patterns()
        self.grammar_patterns = self.get_grammar_patterns()
        self.ollama_config = self.get_ollama_config()
    
    def get_ollama_config(self) -> Dict[str, any]:
        """Get Ollama model configuration for Chinese language processing."""
        return {
            "text_model": {
                "model_id": "qwen2.5:7b",  # Good for Chinese text processing
                "temperature": 0.3,  # Lower temperature for more consistent Chinese output
                "max_tokens": 2000,
                "system_prompt": "你是一个专业的中文文本分析助手，擅长实体识别、关系提取和知识图谱构建。请用中文回答。",
                "keep_alive": "10m"
            },
            "vision_model": {
                "model_id": "llava:latest",  # Good for Chinese OCR and image analysis
                "temperature": 0.4,
                "max_tokens": 1500,
                "system_prompt": "你是一个专业的中文图像分析助手，擅长中文OCR、图像理解和多模态分析。请用中文回答。",
                "keep_alive": "15m"
            },
            "audio_model": {
                "model_id": "llava:latest",  # Same as vision for audio processing
                "temperature": 0.4,
                "max_tokens": 1500,
                "system_prompt": "你是一个专业的中文音频分析助手，擅长中文语音识别和音频内容分析。请用中文回答。",
                "keep_alive": "15m"
            },
            "classical_chinese_model": {
                "model_id": "qwen2.5:7b",  # Good for Classical Chinese
                "temperature": 0.2,  # Very low temperature for Classical Chinese accuracy
                "max_tokens": 2500,
                "system_prompt": "你是一个专业的古汉语分析助手，精通文言文、古典文献和古代文化。请准确识别古汉语中的实体和关系。",
                "keep_alive": "20m"
            }
        }
    
    def get_entity_patterns(self) -> EntityPatterns:
        """Get enhanced Chinese entity patterns including Classical Chinese."""
        return EntityPatterns(
            person=[
                # Modern Chinese names
                r'[\u4e00-\u9fff]{2,4}',  # Chinese names (2-4 characters)
                r'[\u4e00-\u9fff]{2,4}\s+[\u4e00-\u9fff]{2,4}',  # Full names
                r'[\u4e00-\u9fff]{2,4}(?:先生|女士|博士|教授|老师|院士|主席|总理|部长)',  # With titles
                # Classical Chinese names
                r'[\u4e00-\u9fff]{2,4}(?:子|先生|君|公|卿|氏|姓)',  # Classical titles
                r'[\u4e00-\u9fff]{2,4}(?:王|李|张|刘|陈|杨|赵|黄|周|吴)',  # Common surnames
            ],
            organization=[
                # Modern organizations
                r'[\u4e00-\u9fff]+(?:公司|集团|企业|银行|大学|学院|研究所|研究院)',
                r'[\u4e00-\u9fff]+(?:科技|技术|信息|网络|互联网|电子|通信|金融|投资)',
                r'[\u4e00-\u9fff]+(?:政府|部门|部|局|委员会|协会|组织)',
                # Classical organizations
                r'[\u4e00-\u9fff]+(?:国|朝|府|衙|寺|院|馆|阁|楼|台)',  # Classical institutions
                r'[\u4e00-\u9fff]+(?:书院|学堂|私塾|学宫|太学)',  # Classical educational
            ],
            location=[
                # Modern locations
                r'[\u4e00-\u9fff]+(?:市|省|区|县|州|国|地区|城市)',
                r'(?:北京|上海|广州|深圳|杭州|南京|武汉|成都|西安|重庆)',
                r'[\u4e00-\u9fff]+(?:路|街|巷|广场|公园|机场|车站)',
                # Classical locations
                r'[\u4e00-\u9fff]+(?:国|州|郡|县|邑|城|都|京|府|州)',  # Classical administrative
                r'[\u4e00-\u9fff]+(?:山|水|河|江|湖|海|岛|湾|关|塞)',  # Classical geographical
                r'(?:长安|洛阳|开封|临安|金陵|燕京|大都|顺天)',  # Historical capitals
            ],
            concept=[
                # Modern concepts
                r'(?:人工智能|机器学习|深度学习|神经网络|自然语言处理|计算机视觉)',
                r'(?:区块链|云计算|大数据|物联网|量子计算|网络安全|数据科学)',
                r'(?:数字化转型|智能制造|绿色能源|可持续发展|创新技术)',
                # Classical concepts
                r'(?:仁|义|礼|智|信|忠|孝|悌|节|廉)',  # Classical virtues
                r'(?:道|德|理|气|阴阳|五行|八卦|太极|中庸|和谐)',  # Philosophical concepts
                r'(?:诗|书|礼|易|春秋|论语|孟子|大学|中庸)',  # Classical texts
            ]
        )
    
    def get_classical_chinese_patterns(self) -> Dict[str, List[str]]:
        """Get comprehensive Classical Chinese patterns for generic PDF processing."""
        return {
            "particles": [
                r'之|其|者|也|乃|是|于|以|为|所|所以|而|则|故|然|若|虽|但|且|或',
                r'[\u4e00-\u9fff]+(?:之|其|者|也|乃|是)',
                r'(?:何|孰|安|焉|奚|胡|曷|盍|岂|宁|庸|讵)',  # Interrogative particles
                r'(?:夫|盖|惟|唯|独|特|但|然|而|则|故|是以|是故)',  # Classical conjunctions
            ],
            "grammar_structures": [
                r'[\u4e00-\u9fff]+(?:所|所以)[\u4e00-\u9fff]+',  # Nominalization
                r'[\u4e00-\u9fff]+(?:为|被)[\u4e00-\u9fff]+',    # Passive voice
                r'[\u4e00-\u9fff]+(?:以|于)[\u4e00-\u9fff]+',    # Prepositional
                r'[\u4e00-\u9fff]+(?:而|则|故|然)[\u4e00-\u9fff]+',  # Conjunctions
                r'[\u4e00-\u9fff]+(?:者|也|焉|矣|哉|乎|耶|欤)',  # Sentence endings
                r'[\u4e00-\u9fff]+(?:曰|云|谓|言|语|告|对|答)',  # Speech markers
                r'[\u4e00-\u9fff]+(?:见|闻|知|思|想|念|忆|记)',  # Cognitive verbs
            ],
            "classical_entities": [
                r'[\u4e00-\u9fff]{2,4}(?:子|先生|君|公|卿|氏|姓)',  # Classical titles
                r'[\u4e00-\u9fff]+(?:国|州|郡|县|邑|城)',           # Classical locations
                r'(?:仁|义|礼|智|信|忠|孝|悌|节|廉)',              # Classical virtues
                r'(?:道|德|理|气|阴阳|五行)',                      # Philosophical concepts
                r'(?:诗|书|礼|易|春秋|论语|孟子|大学|中庸)',      # Classical texts
                r'[\u4e00-\u9fff]+(?:王|帝|皇|后|妃|太子|公主)',   # Royal titles
                r'[\u4e00-\u9fff]+(?:将军|元帅|将军|校尉|都尉)',   # Military titles
                r'[\u4e00-\u9fff]+(?:丞相|尚书|侍郎|御史|太守)',   # Official titles
            ],
            "measure_words": [
                r'(?:个|只|条|张|片|块|本|册|卷|篇|首|句|字|词)',
                r'(?:丈|尺|寸|里|亩|顷|石|斗|升|斤|两|钱)',
                r'(?:匹|头|口|尾|羽|只|双|对|副|套)',  # Classical measure words
            ],
            "time_expressions": [
                r'(?:春|夏|秋|冬|年|月|日|时|刻|更|夜|晨|暮)',
                r'(?:甲|乙|丙|丁|戊|己|庚|辛|壬|癸)',  # Heavenly stems
                r'(?:子|丑|寅|卯|辰|巳|午|未|申|酉|戌|亥)',  # Earthly branches
                r'[\u4e00-\u9fff]+(?:年|月|日|时|刻|更|夜)',  # Time units
                r'(?:元|明|清|唐|宋|汉|秦|周|商|夏)',  # Dynasties
            ],
            "pdf_generic": [
                r'[\u4e00-\u9fff]+(?:章|节|篇|卷|部|集|册)',  # Chapter markers
                r'[\u4e00-\u9fff]+(?:注|疏|解|释|义|说)',     # Commentary markers
                r'[\u4e00-\u9fff]+(?:序|跋|引|题|记|录)',     # Preface markers
                r'(?:第|初|次|再|又|复|重|新|旧|古|今)',      # Ordinal and temporal
                r'[\u4e00-\u9fff]+(?:页|面|行|段|句|字)',     # Document structure
                r'[\u4e00-\u9fff]+(?:标题|题目|主题|内容|正文)',  # Content markers
                r'[\u4e00-\u9fff]+(?:作者|编者|译者|校者)',   # Author markers
                r'[\u4e00-\u9fff]+(?:出版|发行|印刷|制作)',   # Publication markers
            ]
        }
    
    def get_grammar_patterns(self) -> Dict[str, List[str]]:
        """Get comprehensive grammar patterns for Chinese."""
        return {
            "modern_grammar": [
                r'[\u4e00-\u9fff]+(?:的|地|得)[\u4e00-\u9fff]+',  # Modern particles
                r'[\u4e00-\u9fff]+(?:了|着|过)[\u4e00-\u9fff]*',  # Aspect markers
                r'[\u4e00-\u9fff]+(?:吗|呢|吧|啊|呀|哇)',  # Modal particles
            ],
            "classical_grammar": [
                r'[\u4e00-\u9fff]+(?:之|其|者|也|乃|是)',  # Classical particles
                r'[\u4e00-\u9fff]+(?:所|所以)[\u4e00-\u9fff]+',  # Nominalization
                r'[\u4e00-\u9fff]+(?:为|被)[\u4e00-\u9fff]+',    # Passive voice
            ],
            "sentence_patterns": [
                r'[\u4e00-\u9fff]+(?:是|为|乃)[\u4e00-\u9fff]+',  # Copula patterns
                r'[\u4e00-\u9fff]+(?:有|无|没)[\u4e00-\u9fff]*',  # Existence patterns
                r'[\u4e00-\u9fff]+(?:在|于)[\u4e00-\u9fff]+',    # Location patterns
            ]
        }
    
    def get_processing_settings(self) -> ProcessingSettings:
        """Get enhanced Chinese processing settings for orphan nodes reduction."""
        return ProcessingSettings(
            min_entity_length=2,  # Chinese can have shorter entities
            max_entity_length=20,  # Chinese entities are typically shorter
            confidence_threshold=0.7,
            use_enhanced_extraction=True,
            relationship_prompt_simplified=True,  # Use simplified for Chinese
            use_hierarchical_relationships=True,  # Enable hierarchical relationships
            entity_clustering_enabled=True,  # Enable entity clustering
            fallback_strategies=[
                "hierarchical",
                "proximity", 
                "template",
                "semantic",
                "classical_patterns"  # New fallback strategy
            ]
        )
    
    def get_relationship_templates(self) -> Dict[str, str]:
        """Get enhanced Chinese relationship templates."""
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
            # Classical Chinese relationships
            "classical_person_title": "HAS_TITLE",
            "classical_location_administrative": "ADMINISTERS",
            "classical_concept_philosophy": "REPRESENTS_PHILOSOPHY",
            "classical_text_author": "WRITTEN_BY",
        }
    
    def get_detection_patterns(self) -> List[str]:
        """Get enhanced Chinese language detection patterns."""
        return [
            r'[\u4e00-\u9fff]',  # Chinese characters
            r'\b(?:的|是|在|有|和|与|或|但|而|因为|所以|如果|虽然)\b',  # Common particles
            r'\b(?:中国|中文|汉语|普通话|简体|繁体)\b',  # Chinese language terms
            r'\b(?:你好|谢谢|再见|对不起|没关系|请问|当然)\b',  # Common phrases
            # Classical Chinese detection
            r'\b(?:之|其|者|也|乃|是|于|以|为|所|所以|而|则|故|然)\b',  # Classical particles
            r'\b(?:仁|义|礼|智|信|忠|孝|悌|节|廉)\b',  # Classical virtues
            r'\b(?:道|德|理|气|阴阳|五行)\b',  # Philosophical concepts
        ]
    
    def get_hierarchical_relationship_prompt(self, entities: List[str], text: str) -> str:
        """Get Chinese-specific hierarchical relationship prompt."""
        entities_str = ", ".join(entities[:15])  # More entities for Chinese
        return f"""
分析以下中文文本，识别实体之间的层次关系：{entities_str}

文本：{text[:1500]}...

请识别以下类型的层次关系：
1. 父子关系：上级与下级、包含与被包含
2. 同级关系：同一类别中的实体
3. 位置关系：地理位置、机构位置
4. 功能关系：工作关系、专业领域
5. 概念关系：相关概念、技术关联
6. 古典关系：古典人物、古典机构、古典概念

请按以下格式提供关系：
实体1 | 关系类型 | 实体2

重点关注最重要的关系，确保准确性和相关性。
"""
    
    def get_entity_clustering_prompt(self, entities: List[str], text: str) -> str:
        """Get Chinese-specific entity clustering prompt."""
        entities_str = ", ".join(entities[:20])
        return f"""
将以下中文实体按语义相似性分组：{entities_str}

文本：{text[:1000]}...

请按以下标准分组：
1. 人物组：相关的人物实体（包括古典人物）
2. 机构组：相关的组织机构（包括古典机构）
3. 地点组：相关的地理位置（包括古典地名）
4. 概念组：相关的技术概念（包括古典概念）
5. 古典组：古典文本、古典思想、古典制度

为每个组内的实体创建关系，格式：
实体1 | 同组关系 | 实体2

确保分组合理，关系有意义。
"""
    
    def is_classical_chinese(self, text: str) -> bool:
        """Detect if text contains Classical Chinese patterns."""
        import re
        classical_indicators = [
            r'之|其|者|也|乃|是|于|以|为|所|所以|而|则|故|然',
            r'仁|义|礼|智|信|忠|孝|悌|节|廉',
            r'道|德|理|气|阴阳|五行',
            r'诗|书|礼|易|春秋|论语|孟子|大学|中庸'
        ]
        
        for pattern in classical_indicators:
            if re.search(pattern, text):
                return True
        return False
    
    def get_classical_processing_settings(self) -> ProcessingSettings:
        """Get specialized processing settings for Classical Chinese."""
        return ProcessingSettings(
            min_entity_length=1,  # Classical Chinese can have single character entities
            max_entity_length=15,  # Classical entities are typically shorter
            confidence_threshold=0.8,  # Higher threshold for Classical Chinese
            use_enhanced_extraction=True,
            relationship_prompt_simplified=False,  # Use detailed prompts for Classical Chinese
            use_hierarchical_relationships=True,
            entity_clustering_enabled=True,
            fallback_strategies=[
                "classical_patterns",
                "hierarchical",
                "semantic",
                "template"
            ]
        )
    
    def get_chinese_pdf_processing_settings(self) -> ProcessingSettings:
        """Get specialized processing settings for Chinese PDF documents."""
        return ProcessingSettings(
            min_entity_length=1,  # Chinese PDFs can have single character entities
            max_entity_length=25,  # Chinese PDF entities can be longer
            confidence_threshold=0.75,  # Balanced threshold for Chinese PDFs
            use_enhanced_extraction=True,
            relationship_prompt_simplified=False,  # Use detailed prompts for PDFs
            use_hierarchical_relationships=True,
            entity_clustering_enabled=True,
            fallback_strategies=[
                "pdf_generic",
                "classical_patterns",
                "hierarchical",
                "semantic",
                "template"
            ]
        )
    
    def detect_chinese_pdf_type(self, text: str) -> str:
        """Detect the type of Chinese PDF content for optimal processing."""
        import re
        
        # Check for Classical Chinese indicators
        classical_indicators = [
            r'之|其|者|也|乃|是|于|以|为|所|所以|而|则|故|然',
            r'仁|义|礼|智|信|忠|孝|悌|节|廉',
            r'道|德|理|气|阴阳|五行',
            r'诗|书|礼|易|春秋|论语|孟子|大学|中庸'
        ]
        
        classical_score = 0
        for pattern in classical_indicators:
            if re.search(pattern, text):
                classical_score += 1
        
        # Check for modern Chinese indicators
        modern_indicators = [
            r'的|地|得|了|着|过|吗|呢|吧|啊|呀|哇',
            r'人工智能|机器学习|深度学习|神经网络',
            r'公司|集团|企业|银行|大学|学院',
            r'政府|部门|部|局|委员会|协会'
        ]
        
        modern_score = 0
        for pattern in modern_indicators:
            if re.search(pattern, text):
                modern_score += 1
        
        # Determine PDF type based on scores
        if classical_score > modern_score and classical_score >= 2:
            return "classical_chinese"
        elif modern_score > classical_score and modern_score >= 2:
            return "modern_chinese"
        else:
            return "mixed_chinese"  # Default for mixed or unclear content
