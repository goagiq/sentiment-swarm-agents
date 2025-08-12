"""
Japanese language configuration for enhanced processing.
Comprehensive regex patterns for Japanese entity extraction and grammar optimization.
"""

from typing import Dict, List
from .base_config import BaseLanguageConfig, EntityPatterns, ProcessingSettings


class JapaneseConfig(BaseLanguageConfig):
    """Japanese language configuration with comprehensive regex patterns."""
    
    def __init__(self):
        super().__init__()
        self.language_code = "ja"
        self.language_name = "Japanese"
        self.entity_patterns = self.get_entity_patterns()
        self.processing_settings = self.get_processing_settings()
        self.relationship_templates = self.get_relationship_templates()
        self.detection_patterns = self.get_detection_patterns()
        self.grammar_patterns = self.get_grammar_patterns()
        self.honorific_patterns = self.get_honorific_patterns()
    
    def get_entity_patterns(self) -> EntityPatterns:
        """Get comprehensive Japanese entity patterns."""
        return EntityPatterns(
            person=[
                # Japanese names (Kanji + Hiragana)
                r'[\u4E00-\u9FAF]{2,6}',  # Kanji names
                r'[\u4E00-\u9FAF]+[\u3040-\u309F]+',  # Kanji + Hiragana
                r'[\u3040-\u309F]+[\u4E00-\u9FAF]+',  # Hiragana + Kanji
                # Names with honorifics
                r'[\u4E00-\u9FAF]+(?:さん|様|先生|博士|教授|社長|部長|課長)',
                # Full names with spaces
                r'[\u4E00-\u9FAF]+\s+[\u4E00-\u9FAF]+',
                # Katakana names (foreign names)
                r'[\u30A0-\u30FF]{2,8}',
            ],
            organization=[
                # Company types
                r'[\u4E00-\u9FAF]+(?:株式会社|有限会社|合同会社|合資会社)',
                r'[\u4E00-\u9FAF]+(?:コーポレーション|カンパニー|グループ)',
                # Educational institutions
                r'[\u4E00-\u9FAF]+(?:大学|学院|専門学校|研究所|センター)',
                # Government organizations
                r'[\u4E00-\u9FAF]+(?:省|庁|局|部|課|委員会|議会)',
                # Research institutions
                r'[\u4E00-\u9FAF]+(?:研究所|研究院|ラボラトリー|センター)',
                # Media organizations
                r'[\u4E00-\u9FAF]+(?:放送|テレビ|ラジオ|新聞|出版社)',
            ],
            location=[
                # Administrative divisions
                r'[\u4E00-\u9FAF]+(?:都|道|府|県|市|区|町|村)',
                # Major cities
                r'(?:東京|大阪|名古屋|横浜|神戸|京都|福岡|札幌|仙台|広島)',
                # Geographic features
                r'[\u4E00-\u9FAF]+(?:山|川|湖|海|島|湾|港|空港|駅)',
                # Streets and addresses
                r'[\u4E00-\u9FAF]+(?:通り|丁目|番地|号|階|室)',
                # Historical places
                r'(?:皇居|浅草寺|金閣寺|銀閣寺|清水寺|東大寺)',
            ],
            concept=[
                # Technology concepts
                r'(?:人工知能|機械学習|ディープラーニング|自然言語処理)',
                r'(?:ブロックチェーン|クラウドコンピューティング|ビッグデータ)',
                r'(?:IoT|5G|6G|量子コンピューティング|サイバーセキュリティ)',
                # Business concepts
                r'(?:デジタル変革|DX|サステナビリティ|ESG|イノベーション)',
                r'(?:グローバル化|ローカライゼーション|サプライチェーン)',
                # Cultural concepts
                r'(?:和食|茶道|華道|書道|武道|禅|武士道)',
                r'(?:漫画|アニメ|ゲーム|J-POP|演歌|能|歌舞伎)',
            ]
        )
    
    def get_grammar_patterns(self) -> Dict[str, List[str]]:
        """Get comprehensive grammar patterns for Japanese."""
        return {
            "particles": [
                # Basic particles
                r'(?:は|が|を|に|へ|で|から|まで|より|の|と|や|か|ね|よ|わ)',
                # Compound particles
                r'(?:について|に関して|に対して|によって|として|において)',
                r'(?:として|にとって|について|に関して|に対して|によって)',
            ],
            "verb_forms": [
                # Verb endings
                r'[\u4E00-\u9FAF\u3040-\u309F]+(?:する|なる|いる|ある|れる|られる)',
                r'[\u4E00-\u9FAF\u3040-\u309F]+(?:ます|です|でした|でした|ません|ではありません)',
                # Te-form
                r'[\u4E00-\u9FAF\u3040-\u309F]+(?:て|で|って|んで)',
                # Conditional
                r'[\u4E00-\u9FAF\u3040-\u309F]+(?:ば|たら|なら|と)',
            ],
            "adjectives": [
                # I-adjectives
                r'[\u4E00-\u9FAF\u3040-\u309F]+(?:い|く|くて|かった|くなかった)',
                # Na-adjectives
                r'[\u4E00-\u9FAF\u3040-\u309F]+(?:な|に|で|だった|じゃなかった)',
            ],
            "honorifics": [
                # Polite forms
                r'(?:お|ご)[\u4E00-\u9FAF\u3040-\u309F]+',
                r'[\u4E00-\u9FAF\u3040-\u309F]+(?:様|さん|先生|博士|教授)',
            ],
            "counters": [
                # Common counters
                r'(?:個|枚|本|冊|台|匹|頭|羽|杯|瓶|箱|袋|束|組|対)',
                r'(?:人|名|歳|年|月|日|時|分|秒|回|度|番|階|軒|件)',
            ]
        }
    
    def get_honorific_patterns(self) -> Dict[str, List[str]]:
        """Get Japanese honorific patterns."""
        return {
            "keigo": [
                # Sonkeigo (respectful language)
                r'(?:お|ご)[\u4E00-\u9FAF\u3040-\u309F]+(?:になる|なさる|くださる)',
                r'[\u4E00-\u9FAF\u3040-\u309F]+(?:れる|られる|なさる|くださる)',
                # Kenjougo (humble language)
                r'(?:お|ご)[\u4E00-\u9FAF\u3040-\u309F]+(?:する|いたす|申す)',
                r'[\u4E00-\u9FAF\u3040-\u309F]+(?:いたす|申す|参る|拝見する)',
            ],
            "titles": [
                # Professional titles
                r'(?:社長|部長|課長|主任|係長|主任|担当|責任者)',
                r'(?:教授|准教授|講師|助教|研究員|博士|修士)',
                r'(?:医師|看護師|薬剤師|弁護士|会計士|税理士)',
            ],
            "family_titles": [
                # Family honorifics
                r'(?:お父さん|お母さん|お兄さん|お姉さん|おじいさん|おばあさん)',
                r'(?:父|母|兄|姉|祖父|祖母|叔父|叔母)',
            ]
        }
    
    def get_processing_settings(self) -> ProcessingSettings:
        """Get Japanese processing settings."""
        return ProcessingSettings(
            min_entity_length=2,  # Japanese can have shorter entities
            max_entity_length=25,  # Japanese entities are typically shorter
            confidence_threshold=0.75,
            use_enhanced_extraction=True,
            relationship_prompt_simplified=False,  # Japanese needs detailed prompts
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
        """Get Japanese relationship templates."""
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
            # Japanese-specific relationships
            "person_honorific": "HAS_TITLE",
            "organization_type": "IS_TYPE_OF",
            "location_administrative": "ADMINISTERS",
            "concept_field": "BELONGS_TO_FIELD",
            "family_relationship": "FAMILY_OF",
        }
    
    def get_detection_patterns(self) -> List[str]:
        """Get Japanese language detection patterns."""
        return [
            r'[\u3040-\u309F]',  # Hiragana
            r'[\u30A0-\u30FF]',  # Katakana
            r'[\u4E00-\u9FAF]',  # Kanji
            # Common Japanese words
            r'\b(?:です|ます|でした|でした|ありません|ではありません)\b',
            r'\b(?:は|が|を|に|へ|で|から|まで|より|の|と|や|か)\b',
            r'\b(?:お|ご)[\u4E00-\u9FAF\u3040-\u309F]+\b',  # Honorific prefixes
            r'\b[\u4E00-\u9FAF\u3040-\u309F]+(?:さん|様|先生|博士)\b',  # Honorific suffixes
            # Japanese-specific terms
            r'\b(?:日本|日本語|和|大和|倭|日)\b',
            r'\b(?:こんにちは|ありがとう|すみません|おはよう|こんばんは)\b',
        ]
    
    def get_hierarchical_relationship_prompt(self, entities: List[str], text: str) -> str:
        """Get Japanese-specific hierarchical relationship prompt."""
        entities_str = ", ".join(entities[:15])
        return f"""
以下の日本語テキストを分析し、これらのエンティティ間の階層関係を特定してください：{entities_str}

テキスト：{text[:1500]}...

以下のタイプの階層関係を特定してください：
1. 上下関係：上司と部下、包含と被包含
2. 同レベル関係：同じカテゴリのエンティティ
3. 場所関係：地理的位置、機関の位置
4. 機能関係：仕事関係、専門分野
5. 概念関係：関連概念、技術的関連
6. 敬語関係：敬語表現、社会的地位

以下の形式で関係を提供してください：
エンティティ1 | 関係タイプ | エンティティ2

最も重要な関係に焦点を当て、正確性と関連性を確保してください。
"""
    
    def get_entity_clustering_prompt(self, entities: List[str], text: str) -> str:
        """Get Japanese-specific entity clustering prompt."""
        entities_str = ", ".join(entities[:20])
        return f"""
以下の日本語エンティティを意味的類似性に基づいてグループ化してください：{entities_str}

テキスト：{text[:1000]}...

以下の基準でグループ化してください：
1. 人物グループ：関連する人物エンティティ
2. 組織グループ：関連する組織
3. 場所グループ：関連する地理的位置
4. 概念グループ：関連する技術概念
5. 敬語グループ：敬語表現、社会的地位

各グループ内のエンティティの関係を作成し、形式：
エンティティ1 | グループ内関係 | エンティティ2

合理的なグループ化と意味のある関係を確保してください。
"""
    
    def is_formal_japanese(self, text: str) -> bool:
        """Detect if text contains formal Japanese patterns."""
        import re
        formal_indicators = [
            r'お|ご',  # Honorific prefixes
            r'です|ます|でした|でした|ありません|ではありません',  # Polite forms
            r'様|さん|先生|博士|教授|社長|部長|課長',  # Honorific suffixes
            r'いたします|申します|参ります|拝見します',  # Humble forms
            r'なさいます|くださいます|おっしゃいます',  # Respectful forms
        ]
        
        for pattern in formal_indicators:
            if re.search(pattern, text):
                return True
        return False
    
    def get_formal_processing_settings(self) -> ProcessingSettings:
        """Get specialized processing settings for formal Japanese."""
        return ProcessingSettings(
            min_entity_length=2,
            max_entity_length=30,  # Formal Japanese can have longer entities
            confidence_threshold=0.8,  # Higher threshold for formal Japanese
            use_enhanced_extraction=True,
            relationship_prompt_simplified=False,  # Use detailed prompts for formal Japanese
            use_hierarchical_relationships=True,
            entity_clustering_enabled=True,
            fallback_strategies=[
                "honorific_patterns",
                "formal_patterns",
                "grammar_patterns",
                "hierarchical",
                "semantic"
            ]
        )
