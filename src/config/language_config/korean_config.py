"""
Korean language configuration for enhanced processing.
Comprehensive regex patterns for Korean entity extraction and grammar optimization.
"""

from typing import Dict, List
from .base_config import BaseLanguageConfig, EntityPatterns, ProcessingSettings


class KoreanConfig(BaseLanguageConfig):
    """Korean language configuration with comprehensive regex patterns."""
    
    def __init__(self):
        super().__init__()
        self.language_code = "ko"
        self.language_name = "Korean"
        self.entity_patterns = self.get_entity_patterns()
        self.processing_settings = self.get_processing_settings()
        self.relationship_templates = self.get_relationship_templates()
        self.detection_patterns = self.get_detection_patterns()
        self.grammar_patterns = self.get_grammar_patterns()
        self.honorific_patterns = self.get_honorific_patterns()
    
    def get_entity_patterns(self) -> EntityPatterns:
        """Get comprehensive Korean entity patterns."""
        return EntityPatterns(
            person=[
                # Korean names
                r'[가-힣]{2,4}',  # Korean names
                r'[가-힣]+(?:씨|님|선생님|박사|교수|사장|부장|과장)',
                # Full names with spaces
                r'[가-힣]+\s+[가-힣]+',
                # Names with titles
                r'(?:김|이|박|최|정|강|조|윤|장|임)\s*[가-힣]+',
            ],
            organization=[
                # Company types
                r'[가-힣]+(?:주식회사|유한회사|합자회사|합명회사)',
                r'[가-힣]+(?:기업|그룹|컴퍼니|코퍼레이션)',
                # Educational institutions
                r'[가-힣]+(?:대학교|대학원|전문학교|연구소|센터)',
                # Government organizations
                r'[가-힣]+(?:부|청|국|과|팀|위원회|의회)',
                # Research institutions
                r'[가-힣]+(?:연구원|연구소|연구기관|센터)',
            ],
            location=[
                # Administrative divisions
                r'[가-힣]+(?:시|도|군|구|동|읍|면|리)',
                # Major cities
                r'(?:서울|부산|대구|인천|광주|대전|울산|세종)',
                # Geographic features
                r'[가-힣]+(?:산|강|호|바다|섬|만|항|공항|역)',
                # Streets and addresses
                r'[가-힣]+(?:로|길|동|번지|호|층|실)',
            ],
            concept=[
                # Technology concepts
                r'(?:인공지능|머신러닝|딥러닝|자연어처리)',
                r'(?:블록체인|클라우드컴퓨팅|빅데이터|사물인터넷)',
                r'(?:5G|6G|양자컴퓨팅|사이버보안)',
                # Business concepts
                r'(?:디지털전환|DX|지속가능성|ESG|혁신)',
                r'(?:글로벌화|로컬라이제이션|공급망)',
                # Cultural concepts
                r'(?:한식|차문화|꽃문화|서예|무도|선|무사도)',
                r'(?:만화|애니메이션|게임|K-POP|트로트)',
            ]
        )
    
    def get_grammar_patterns(self) -> Dict[str, List[str]]:
        """Get comprehensive grammar patterns for Korean."""
        return {
            "particles": [
                r'(?:은|는|이|가|을|를|에|에서|로|으로|의|와|과|하고|며|면서)',
                r'(?:부터|까지|보다|처럼|같이|만|도|까지|마다|마다)',
            ],
            "verb_endings": [
                r'[가-힣]+(?:다|니다|습니다|어요|아요|어|아|고|며|면서)',
                r'[가-힣]+(?:ㄴ다|는다|는다|는다|는다|는다)',
            ],
            "honorifics": [
                r'(?:시|님|씨|선생님|박사|교수)',
                r'[가-힣]+(?:시|님|씨|선생님|박사|교수)',
            ],
            "counters": [
                r'(?:개|명|마리|권|대|개|장|벌|켤레|쌍|세트)',
                r'(?:명|분|살|년|월|일|시|분|초|번|회|차)',
            ]
        }
    
    def get_honorific_patterns(self) -> Dict[str, List[str]]:
        """Get Korean honorific patterns."""
        return {
            "formal_speech": [
                r'[가-힣]+(?:습니다|니다|입니다|습니다|습니다)',
                r'[가-힣]+(?:시|님|씨|선생님|박사|교수)',
            ],
            "titles": [
                r'(?:사장|부장|과장|대리|주임|사원)',
                r'(?:교수|박사|석사|학사|연구원|조교)',
                r'(?:의사|간호사|약사|변호사|회계사|세무사)',
            ]
        }
    
    def get_processing_settings(self) -> ProcessingSettings:
        """Get Korean processing settings."""
        return ProcessingSettings(
            min_entity_length=2,
            max_entity_length=20,
            confidence_threshold=0.75,
            use_enhanced_extraction=True,
            relationship_prompt_simplified=False,
            use_hierarchical_relationships=True,
            entity_clustering_enabled=True,
            fallback_strategies=[
                "honorific_patterns",
                "grammar_patterns",
                "hierarchical",
                "semantic",
                "template"
            ]
        )
    
    def get_relationship_templates(self) -> Dict[str, str]:
        """Get Korean relationship templates."""
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
            "person_honorific": "HAS_TITLE",
            "organization_type": "IS_TYPE_OF",
        }
    
    def get_detection_patterns(self) -> List[str]:
        """Get Korean language detection patterns."""
        return [
            r'[가-힣]',  # Korean characters
            r'\b(?:입니다|습니다|어요|아요|다|네|요)\b',
            r'\b(?:은|는|이|가|을|를|에|에서|로|으로|의|와|과)\b',
            r'\b(?:안녕하세요|감사합니다|죄송합니다|네|아니요)\b',
            r'\b(?:한국|한국어|한글|대한민국|조선)\b',
        ]
    
    def is_formal_korean(self, text: str) -> bool:
        """Detect if text contains formal Korean patterns."""
        import re
        formal_indicators = [
            r'습니다|니다|입니다',
            r'시|님|씨|선생님',
            r'감사합니다|죄송합니다|안녕하세요',
        ]
        
        for pattern in formal_indicators:
            if re.search(pattern, text):
                return True
        return False
