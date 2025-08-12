"""
Arabic language configuration for enhanced processing.
Enhanced with comprehensive regex patterns and grammar optimization.
"""

from typing import Dict, List
from .base_config import BaseLanguageConfig, EntityPatterns, ProcessingSettings


class ArabicConfig(BaseLanguageConfig):
    """Arabic language configuration with enhanced regex patterns and grammar support."""
    
    def __init__(self):
        super().__init__()
        self.language_code = "ar"
        self.language_name = "Arabic"
        self.entity_patterns = self.get_entity_patterns()
        self.processing_settings = self.get_processing_settings()
        self.relationship_templates = self.get_relationship_templates()
        self.detection_patterns = self.get_detection_patterns()
        self.grammar_patterns = self.get_grammar_patterns()
        self.advanced_patterns = self.get_advanced_patterns()
    
    def get_entity_patterns(self) -> EntityPatterns:
        """Get enhanced Arabic entity patterns."""
        return EntityPatterns(
            person=[
                # Arabic names with various forms
                r'[\u0600-\u06FF]{2,4}\s+[\u0600-\u06FF]{2,4}(?:\s+[\u0600-\u06FF]{2,4})?',  # Full names
                r'[\u0600-\u06FF]{2,4}\s+[\u0600-\u06FF]{2,4}(?:بن|بنت)\s+[\u0600-\u06FF]{2,4}',  # Names with "son of/daughter of"
                r'[\u0600-\u06FF]{2,4}(?:أبو|أم)\s+[\u0600-\u06FF]{2,4}',  # Names with "father of/mother of"
                # Names with titles
                r'(?:الشيخ|الدكتور|الأستاذ|البروفيسور|المهندس|المحامي)\s+[\u0600-\u06FF]{2,4}',
                r'[\u0600-\u06FF]{2,4}\s+(?:الشيخ|الدكتور|الأستاذ|البروفيسور|المهندس|المحامي)',
                # Common Arabic names
                r'(?:محمد|أحمد|علي|عبدالله|عبدالرحمن|عبدالعزيز|عبداللطيف|عبدالمجيد)',
                r'(?:فاطمة|عائشة|خديجة|مريم|زينب|نور|سارة|ليلى|رنا|نورا)',
            ],
            organization=[
                # Business organizations
                r'[\u0600-\u06FF]+(?:شركة|مؤسسة|مجموعة|بنك|مصرف|استثمار|تطوير)',
                r'[\u0600-\u06FF]+(?:للصناعة|للتجارة|للاستثمار|للتطوير|للتكنولوجيا)',
                # Educational institutions
                r'[\u0600-\u06FF]+(?:جامعة|كلية|معهد|أكاديمية|مدرسة|مركز)',
                r'[\u0600-\u06FF]+(?:للتعليم|للبحث|للدراسات|للتطوير)',
                # Government organizations
                r'[\u0600-\u06FF]+(?:وزارة|هيئة|إدارة|مكتب|لجنة|مجلس)',
                r'[\u0600-\u06FF]+(?:الحكومية|الوطنية|المحلية|الإقليمية)',
                # Media organizations
                r'[\u0600-\u06FF]+(?:قناة|إذاعة|صحيفة|مجلة|وكالة|مؤسسة إعلامية)',
                # Religious organizations
                r'[\u0600-\u06FF]+(?:مسجد|جامع|مدرسة|مركز إسلامي|جمعية خيرية)',
            ],
            location=[
                # Countries and cities
                r'(?:مصر|السعودية|الإمارات|الكويت|قطر|البحرين|عمان|الأردن|لبنان|سوريا|العراق|اليمن)',
                r'(?:القاهرة|الرياض|جدة|دبي|أبوظبي|الكويت|الدوحة|المنامة|مسقط|عمان|بيروت|دمشق|بغداد|صنعاء)',
                # Administrative divisions
                r'[\u0600-\u06FF]+(?:محافظة|مديرية|قضاء|ناحية|حي|شارع|ميدان|ساحة)',
                # Geographic features
                r'[\u0600-\u06FF]+(?:جبل|وادي|نهر|بحر|خليج|جزيرة|صحراء|واحة)',
                # Historical places
                r'(?:الأزهر|الحرم المكي|الحرم النبوي|المسجد الأقصى|قبة الصخرة)',
            ],
            concept=[
                # Technology concepts
                r'(?:الذكاء الاصطناعي|التعلم الآلي|التعلم العميق|الشبكات العصبية)',
                r'(?:سلسلة الكتل|الحوسبة السحابية|البيانات الضخمة|إنترنت الأشياء)',
                r'(?:الحوسبة الكمية|الأمن السيبراني|علم البيانات|الروبوتات)',
                # Economic concepts
                r'(?:الاقتصاد الرقمي|التحول الرقمي|التجارة الإلكترونية|التمويل التقني)',
                r'(?:الاقتصاد الأخضر|التنمية المستدامة|الطاقة المتجددة|الاقتصاد المعرفي)',
                # Cultural concepts
                r'(?:الأدب العربي|الشعر العربي|الفن الإسلامي|العمارة الإسلامية)',
                r'(?:الفلسفة الإسلامية|علم الكلام|التصوف|الفقه الإسلامي)',
                # Scientific concepts
                r'(?:الفيزياء الكمية|البيولوجيا الجزيئية|الفضاء|الطب الحديث)',
            ]
        )
    
    def get_grammar_patterns(self) -> Dict[str, List[str]]:
        """Get comprehensive grammar patterns for Arabic."""
        return {
            "definite_article": [
                r'ال[\u0600-\u06FF]+',  # Words with definite article
                r'[\u0600-\u06FF]+ال[\u0600-\u06FF]+',  # Words with definite article in middle
            ],
            "verb_forms": [
                r'[\u0600-\u06FF]+(?:ي|ت|ن|أ|نحن|أنتم|هم|هن)',  # Verb prefixes
                r'[\u0600-\u06FF]+(?:ت|تما|تم|تن|تنا|تم|تن)',  # Verb suffixes
                r'[\u0600-\u06FF]+(?:وا|وا|ن|ت|تما|تم|تن)',  # Verb endings
            ],
            "prepositions": [
                r'\b(?:في|على|إلى|من|عن|مع|ب|ل|ك|حول|خلف|أمام|تحت|فوق|بين|وسط)\b',
                r'\b(?:بسبب|بفضل|بعد|قبل|خلال|أثناء|منذ|حتى|إلى|عند|عندما)\b',
            ],
            "conjunctions": [
                r'\b(?:و|أو|لكن|إلا|بل|أما|إذ|إذا|لأن|لكي|حيث|التي|الذي|الذين)\b',
                r'\b(?:عندما|بينما|بعدما|قبلما|حيثما|أينما|كيفما|متى)\b',
            ],
            "question_words": [
                r'\b(?:ما|من|أين|متى|كيف|لماذا|أي|كم|أيها|أيتها)\b',
                r'\b(?:هل|أ|أما|أم|أو|ألا|أليس|أليست)\b',
            ],
            "numbers": [
                r'\b(?:صفر|واحد|اثنان|ثلاثة|أربعة|خمسة|ستة|سبعة|ثمانية|تسعة|عشرة)\b',
                r'\b(?:أول|ثاني|ثالث|رابع|خامس|سادس|سابع|ثامن|تاسع|عاشر)\b',
            ]
        }
    
    def get_advanced_patterns(self) -> Dict[str, List[str]]:
        """Get advanced Arabic patterns for specialized processing."""
        return {
            "islamic_terms": [
                r'(?:الله|الرحمن|الرحيم|السلام|الإسلام|المسلم|المسلمة)',
                r'(?:القرآن|الحديث|السنة|الفقه|الشريعة|الحلال|الحرام)',
                r'(?:الصلاة|الصوم|الزكاة|الحج|العمرة|الجهاد|الدعاء)',
            ],
            "arabic_calligraphy": [
                r'[\u0600-\u06FF]{2,}(?:خط|كتابة|فن|تصميم|زخرفة)',
                r'(?:الخط العربي|الخط الكوفي|الخط النسخ|الخط الثلث|الخط الديواني)',
            ],
            "arabic_poetry": [
                r'(?:شعر|قصيدة|بيت|قافية|وزن|عروض|بحور|أوزان)',
                r'(?:العمود|التفعيلة|النثر|الموشح|الزجل|الكان كان)',
            ],
            "arabic_philosophy": [
                r'(?:فلسفة|منطق|عقل|نفس|روح|وجود|جوهر|عرض)',
                r'(?:ابن سينا|ابن رشد|الفارابي|الكندي|الرازي|الغزالي)',
            ],
            "arabic_science": [
                r'(?:رياضيات|هندسة|طب|صيدلة|فلك|كيمياء|فيزياء|جغرافيا)',
                r'(?:الخوارزمي|ابن الهيثم|الرازي|ابن النفيس|البيروني)',
            ]
        }
    
    def get_processing_settings(self) -> ProcessingSettings:
        """Get Arabic-specific processing settings."""
        return ProcessingSettings(
            min_entity_length=2,  # Arabic can have shorter entities
            max_entity_length=30,  # Arabic entities can be longer
            confidence_threshold=0.75,
            use_enhanced_extraction=True,
            relationship_prompt_simplified=False,  # Arabic needs detailed prompts
            use_hierarchical_relationships=True,
            entity_clustering_enabled=True,
            fallback_strategies=[
                "arabic_patterns",
                "islamic_terms",
                "hierarchical",
                "semantic",
                "template"
            ]
        )
    
    def get_relationship_templates(self) -> Dict[str, List[str]]:
        """Get Arabic relationship templates."""
        return {
            "person_organization": [
                "يعمل في {person} في {organization}",
                "{person} مدير {organization}",
                "{person} رئيس {organization}",
                "{person} عضو في {organization}",
            ],
            "person_location": [
                "يعيش {person} في {location}",
                "{person} من {location}",
                "{person} مقيم في {location}",
                "{person} ولد في {location}",
            ],
            "organization_location": [
                "يقع {organization} في {location}",
                "{organization} مقرها في {location}",
                "{organization} فرع في {location}",
                "{organization} تأسست في {location}",
            ],
            "concept_related": [
                "{concept1} مرتبط بـ {concept2}",
                "{concept1} يؤدي إلى {concept2}",
                "{concept1} جزء من {concept2}",
                "{concept1} يعتمد على {concept2}",
            ]
        }
    
    def get_detection_patterns(self) -> List[str]:
        """Get Arabic language detection patterns."""
        return [
            r'[\u0600-\u06FF]+',  # Arabic Unicode range
            r'[\u0750-\u077F]+',  # Arabic Supplement
            r'[\u08A0-\u08FF]+',  # Arabic Extended-A
            r'[\uFB50-\uFDFF]+',  # Arabic Presentation Forms-A
            r'[\uFE70-\uFEFF]+',  # Arabic Presentation Forms-B
            r'\b(?:في|على|إلى|من|عن|مع|ب|ل|ك|هذا|هذه|ذلك|تلك|هؤلاء|أولئك)\b',
            r'\b(?:الذي|التي|الذين|اللاتي|اللائي|اللذان|اللتان)\b',
            r'\b(?:و|أو|لكن|إلا|بل|أما|إذ|إذا|لأن|لكي|حيث)\b',
            r'\b(?:إيش|شلون|وين|كيف|أش|إي|أيوا|لا|إيه)\b',  # Gulf dialect
            r'\b(?:إيه|إزاي|فين|عايز|مش|مفيش|كده|كذا)\b',  # Egyptian dialect
            r'\b(?:شو|كيفك|وينك|شلونك|أي|إيوا|لا|إيه)\b',  # Levantine dialect
        ]
