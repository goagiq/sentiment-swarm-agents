"""
Language-specific regex configuration for PDF processing and entity extraction.
This file stores language-specific patterns, regex rules, and processing settings
to handle different languages properly without conflicts.
"""

from typing import Dict, List, Any
import re


# Enhanced Language-specific regex patterns for entity extraction (Phase 3)
LANGUAGE_REGEX_PATTERNS = {
    "en": {
        "person": [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last
            r'\b[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z]\.\b',  # First M. L.
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Middle Last
            r'\b(?:Dr\.|Prof\.|Mr\.|Mrs\.|Ms\.|Sir|Lady|Lord)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # With titles
            r'\b[A-Z][a-z]+(?:-[A-Z][a-z]+)*\s+[A-Z][a-z]+\b',  # Hyphenated names
        ],
        "organization": [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+Inc\.|\s+Corp\.|\s+Ltd\.|\s+LLC|\s+Company|\s+Corporation)\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+University|\s+College|\s+Institute|\s+School|\s+Academy)\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+Government|\s+Agency|\s+Department|\s+Ministry|\s+Office)\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+Foundation|\s+Association|\s+Society|\s+Organization)\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+Research|\s+Laboratory|\s+Center|\s+Institute)\b',
        ],
        "location": [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+City|\s+Town|\s+Village|\s+County|\s+State|\s+Country)\b',
            r'\b(?:New York|Los Angeles|Chicago|Houston|Phoenix|Philadelphia|San Antonio|San Diego|Dallas|San Jose)\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+Street|\s+Avenue|\s+Boulevard|\s+Road|\s+Lane)\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+Park|\s+Square|\s+Plaza|\s+Center|\s+Mall)\b',
        ],
        "concept": [
            r'\b(?:artificial intelligence|machine learning|deep learning|neural networks)\b',
            r'\b(?:blockchain|cloud computing|big data|internet of things|IoT)\b',
            r'\b(?:quantum computing|cybersecurity|data science|robotics)\b',
            r'\b(?:sustainable development|green energy|renewable resources|climate change)\b',
            r'\b(?:digital transformation|smart cities|e-commerce|fintech)\b',
        ]
    },
    
    "zh": {
        "person": [
            r'[\u4e00-\u9fff]{2,4}',  # Chinese names (2-4 characters)
            r'[\u4e00-\u9fff]{2,4}\s+[\u4e00-\u9fff]{2,4}',  # Full names
            r'[\u4e00-\u9fff]{2,4}(?:先生|女士|博士|教授|老师|院士|主席|总理|部长)',  # With titles
            # Classical Chinese names
            r'[\u4e00-\u9fff]{2,4}(?:子|先生|君|公|卿|氏|姓)',  # Classical titles
            r'[\u4e00-\u9fff]{2,4}(?:王|李|张|刘|陈|杨|赵|黄|周|吴)',  # Common surnames
        ],
        "organization": [
            r'[\u4e00-\u9fff]+(?:公司|集团|企业|银行|大学|学院|研究所|研究院)',
            r'[\u4e00-\u9fff]+(?:科技|技术|信息|网络|互联网|电子|通信|金融|投资)',
            r'[\u4e00-\u9fff]+(?:政府|部门|部|局|委员会|协会|组织)',
            # Classical organizations
            r'[\u4e00-\u9fff]+(?:国|朝|府|衙|寺|院|馆|阁|楼|台)',  # Classical institutions
            r'[\u4e00-\u9fff]+(?:书院|学堂|私塾|学宫|太学)',  # Classical educational
        ],
        "location": [
            r'[\u4e00-\u9fff]+(?:市|省|区|县|州|国|地区|城市)',
            r'(?:北京|上海|广州|深圳|杭州|南京|武汉|成都|西安|重庆)',
            r'[\u4e00-\u9fff]+(?:路|街|巷|广场|公园|机场|车站)',
            # Classical locations
            r'[\u4e00-\u9fff]+(?:国|州|郡|县|邑|城|都|京|府|州)',  # Classical administrative
            r'[\u4e00-\u9fff]+(?:山|水|河|江|湖|海|岛|湾|关|塞)',  # Classical geographical
            r'(?:长安|洛阳|开封|临安|金陵|燕京|大都|顺天)',  # Historical capitals
        ],
        "concept": [
            r'(?:人工智能|机器学习|深度学习|神经网络|自然语言处理|计算机视觉)',
            r'(?:区块链|云计算|大数据|物联网|量子计算|网络安全|数据科学)',
            r'(?:数字化转型|智能制造|绿色能源|可持续发展|创新技术)',
            # Classical concepts
            r'(?:仁|义|礼|智|信|忠|孝|悌|节|廉)',  # Classical virtues
            r'(?:道|德|理|气|阴阳|五行|八卦|太极|中庸|和谐)',  # Philosophical concepts
            r'(?:诗|书|礼|易|春秋|论语|孟子|大学|中庸)',  # Classical texts
        ]
    },
    
    "ru": {
        "person": [
            r'\b[А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,})?\b',  # Full names (3+ chars each)
            r'\b[А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.\b',  # Name with initials
            r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,})*\s+(?:господин|госпожа|доктор|профессор)\b',  # With titles
            # Patronymics
            r'\b[А-ЯЁ][а-яё]+(?:ович|евич|овна|евна)\b',
            # Academic titles
            r'\b(?:доктор|профессор|академик|кандидат)\s+[А-ЯЁ][а-яё]+\b',
            # Government titles
            r'\b(?:президент|министр|губернатор|мэр)\s+[А-ЯЁ][а-яё]+\b',
        ],
        "organization": [
            r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[а-яё]{2,})*(?:ООО|ОАО|ЗАО|ПАО|ГК|Корпорация|Компания)\b',
            r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[а-яё]{2,})*(?:Университет|Институт|Академия|Университет)\b',
            r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[а-яё]{2,})*(?:Правительство|Министерство|Агентство)\b',
            # Research institutions
            r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[а-яё]{2,})*(?:НИИ|Институт|Лаборатория|Центр)\b',
            # Media organizations
            r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[а-яё]{2,})*(?:Телеканал|Радио|Газета|Журнал)\b',
        ],
        "location": [
            r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[а-яё]{2,})*(?:город|область|край|республика|район)\b',
            r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[а-яё]{2,})*(?:улица|проспект|переулок|площадь)\b',
            r'\b(?:Москва|Санкт-Петербург|Новосибирск|Екатеринбург|Казань|Россия)\b',  # Major cities
            # Geographic features
            r'\b[А-ЯЁ][а-яё]{2,}(?:\s+[а-яё]{2,})*(?:река|озеро|гора|море|остров)\b',
            # Historical places
            r'\b(?:Кремль|Красная площадь|Эрмитаж|Третьяковская галерея)\b',
        ],
        "concept": [
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
    },
    
    "ja": {
        "person": [
            r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]{2,6}',  # Japanese names
            r'[\u4E00-\u9FAF]+(?:さん|様|先生|博士|教授|社長|部長)',  # Honorifics
            r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+(?:氏|君|ちゃん|くん)',  # Informal titles
        ],
        "organization": [
            r'[\u4E00-\u9FAF]+(?:株式会社|有限会社|合同会社|一般社団法人|公益社団法人)',  # Company types
            r'[\u4E00-\u9FAF]+(?:大学|学院|研究所|センター|財団|協会)',  # Institutions
            r'[\u4E00-\u9FAF]+(?:政府|省|庁|局|部|課|委員会)',  # Government
        ],
        "location": [
            r'[\u4E00-\u9FAF]+(?:県|市|区|町|村|都|府)',  # Administrative divisions
            r'(?:東京|大阪|名古屋|横浜|神戸|京都|福岡|札幌|仙台|広島)',  # Major cities
            r'[\u4E00-\u9FAF]+(?:通り|丁目|番地|駅|空港|港)',  # Addresses
        ],
        "concept": [
            r'(?:人工知能|機械学習|ディープラーニング|自然言語処理)',  # Technology
            r'(?:ブロックチェーン|クラウドコンピューティング|ビッグデータ|IoT)',  # Modern tech
            r'(?:持続可能な開発|環境保護|再生可能エネルギー|気候変動)',  # Environment
        ]
    },
    
    "ko": {
        "person": [
            r'[가-힣]{2,4}',  # Korean names
            r'[가-힣]+(?:씨|님|선생님|박사|교수|사장|부장)',  # Honorifics
            r'[가-힣]+(?:군|양|아|야)',  # Informal titles
        ],
        "organization": [
            r'[가-힣]+(?:주식회사|유한회사|합자회사|공사|재단|협회)',  # Company types
            r'[가-힣]+(?:대학교|대학원|연구소|센터|기관|단체)',  # Institutions
            r'[가-힣]+(?:정부|부|청|국|과|팀|위원회)',  # Government
        ],
        "location": [
            r'[가-힣]+(?:시|군|구|동|읍|면|도)',  # Administrative divisions
            r'(?:서울|부산|대구|인천|광주|대전|울산|세종)',  # Major cities
            r'[가-힣]+(?:로|길|동|번지|역|공항|항구)',  # Addresses
        ],
        "concept": [
            r'(?:인공지능|기계학습|딥러닝|자연어처리)',  # Technology
            r'(?:블록체인|클라우드컴퓨팅|빅데이터|사물인터넷)',  # Modern tech
            r'(?:지속가능한발전|환경보호|재생에너지|기후변화)',  # Environment
        ]
    },
    
    "ar": {
        "person": [
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
        "organization": [
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
        "location": [
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
        "concept": [
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
    },
    
    "hi": {
        "person": [
            # Hindi names with various forms
            r'[\u0900-\u097F]{2,4}\s+[\u0900-\u097F]{2,4}(?:\s+[\u0900-\u097F]{2,4})?',  # Full names
            r'[\u0900-\u097F]{2,4}\s+[\u0900-\u097F]{2,4}(?:सिंह|कुमार|देव|शर्मा|वर्मा|गुप्ता)',  # Common surnames
            # Names with titles
            r'(?:श्री|श्रीमती|डॉ\.|प्रोफेसर|मिस्टर|मिस|कुमारी)\s+[\u0900-\u097F]{2,4}',
            r'[\u0900-\u097F]{2,4}\s+(?:श्री|श्रीमती|डॉ\.|प्रोफेसर|मिस्टर|मिस|कुमारी)',
            # Common Hindi names
            r'(?:राजेश|अमित|प्रवीण|संजय|राहुल|अजय|विकास|दीपक|मनोज|सुरेश)',
            r'(?:प्रिया|कविता|सुनीता|राधा|मीना|रेखा|अंजलि|पूजा|नीतू|रश्मि)',
        ],
        "organization": [
            # Business organizations
            r'[\u0900-\u097F]+(?:कंपनी|लिमिटेड|प्राइवेट|पब्लिक|कॉर्पोरेशन|ग्रुप)',
            r'[\u0900-\u097F]+(?:इंडस्ट्रीज|टेक्नोलॉजी|सॉल्यूशंस|सर्विसेज|ट्रेडिंग)',
            # Educational institutions
            r'[\u0900-\u097F]+(?:विश्वविद्यालय|कॉलेज|स्कूल|इंस्टिट्यूट|अकादमी|महाविद्यालय)',
            r'[\u0900-\u097F]+(?:शिक्षा|शिक्षण|प्रशिक्षण|अनुसंधान|विकास)',
            # Government organizations
            r'[\u0900-\u097F]+(?:सरकार|मंत्रालय|विभाग|कार्यालय|समिति|परिषद)',
            r'[\u0900-\u097F]+(?:राज्य|केंद्र|जिला|ग्राम|नगर|महानगर)',
            # Media organizations
            r'[\u0900-\u097F]+(?:चैनल|रेडियो|समाचार|पत्रिका|प्रकाशन|मीडिया)',
            # Religious organizations
            r'[\u0900-\u097F]+(?:मंदिर|गुरुद्वारा|मस्जिद|चर्च|आश्रम|मठ)',
        ],
        "location": [
            # States and cities
            r'(?:दिल्ली|मुंबई|कोलकाता|चेन्नई|बैंगलोर|हैदराबाद|अहमदाबाद|पुणे|जयपुर|लखनऊ)',
            r'(?:उत्तर प्रदेश|महाराष्ट्र|बिहार|पश्चिम बंगाल|मध्य प्रदेश|तमिलनाडु|राजस्थान|कर्नाटक|गुजरात|आंध्र प्रदेश)',
            # Administrative divisions
            r'[\u0900-\u097F]+(?:राज्य|जिला|तहसील|ब्लॉक|ग्राम|नगर|महानगर|मेट्रो)',
            # Geographic features
            r'[\u0900-\u097F]+(?:पर्वत|पहाड़|नदी|झील|समुद्र|खाड़ी|द्वीप|मरुस्थल|घाटी)',
            # Historical places
            r'(?:ताज महल|लाल किला|गेटवे ऑफ इंडिया|इंडिया गेट|कुतुब मीनार)',
        ],
        "concept": [
            # Technology concepts
            r'(?:कृत्रिम बुद्धिमत्ता|मशीन लर्निंग|डीप लर्निंग|न्यूरल नेटवर्क)',
            r'(?:ब्लॉकचेन|क्लाउड कंप्यूटिंग|बिग डेटा|इंटरनेट ऑफ थिंग्स)',
            r'(?:क्वांटम कंप्यूटिंग|साइबर सुरक्षा|डेटा साइंस|रोबोटिक्स)',
            # Economic concepts
            r'(?:डिजिटल इकोनॉमी|डिजिटल ट्रांसफॉर्मेशन|ई-कॉमर्स|फिनटेक)',
            r'(?:ग्रीन इकोनॉमी|सतत विकास|नवीकरणीय ऊर्जा|ज्ञान अर्थव्यवस्था)',
            # Cultural concepts
            r'(?:हिंदी साहित्य|भारतीय संस्कृति|योग|आयुर्वेद|भारतीय संगीत)',
            r'(?:भारतीय दर्शन|वेद|उपनिषद|गीता|रामायण|महाभारत)',
            # Scientific concepts
            r'(?:क्वांटम भौतिकी|आणविक जीवविज्ञान|अंतरिक्ष विज्ञान|आधुनिक चिकित्सा)',
        ]
    }
}

# Enhanced Language-specific processing settings (Phase 3)
LANGUAGE_PROCESSING_SETTINGS = {
    "en": {
        "min_entity_length": 2,
        "max_entity_length": 50,
        "confidence_threshold": 0.6,
        "use_enhanced_extraction": True,
        "relationship_prompt_simplified": False,
    },
    "zh": {
        "min_entity_length": 2,
        "max_entity_length": 20,
        "confidence_threshold": 0.7,
        "use_enhanced_extraction": True,
        "relationship_prompt_simplified": True,  # Use simplified prompt for Chinese like Russian
    },
    "ru": {
        "min_entity_length": 3,  # Increased minimum length for Russian
        "max_entity_length": 50,
        "confidence_threshold": 0.7,
        "use_enhanced_extraction": True,
        "relationship_prompt_simplified": True,  # Use simplified prompt for Russian
    },
    "ja": {
        "min_entity_length": 2,
        "max_entity_length": 15,
        "confidence_threshold": 0.7,
        "use_enhanced_extraction": True,
        "relationship_prompt_simplified": True,  # Use simplified prompt for Japanese
    },
    "ko": {
        "min_entity_length": 2,
        "max_entity_length": 15,
        "confidence_threshold": 0.7,
        "use_enhanced_extraction": True,
        "relationship_prompt_simplified": True,  # Use simplified prompt for Korean
    },
    "ar": {
        "min_entity_length": 2,
        "max_entity_length": 25,
        "confidence_threshold": 0.7,
        "use_enhanced_extraction": True,
        "relationship_prompt_simplified": True,  # Use simplified prompt for Arabic
    },
    "hi": {
        "min_entity_length": 2,
        "max_entity_length": 25,
        "confidence_threshold": 0.7,
        "use_enhanced_extraction": True,
        "relationship_prompt_simplified": True,  # Use simplified prompt for Hindi
    }
}

# Language-specific relationship mapping prompts (simplified for better JSON parsing)
LANGUAGE_RELATIONSHIP_PROMPTS = {
    "en": """
You are an expert relationship extraction system. Analyze relationships between entities in the given text.

Instructions:
1. Identify significant relationships between the provided entities
2. For each relationship specify:
   - source: Source entity name (exact match from entity list)
   - target: Target entity name (exact match from entity list)
   - relationship_type: Relationship type from [IS_A, PART_OF, LOCATED_IN, WORKS_FOR, CREATED_BY, USES, IMPLEMENTS, SIMILAR_TO, OPPOSES, SUPPORTS, LEADS_TO, DEPENDS_ON, RELATED_TO]
   - confidence: Score from 0.0 to 1.0 based on your confidence
   - description: Clear description of the relationship
3. Include only relationships that are explicitly mentioned or strongly implied in the text
4. Return ONLY valid JSON in the specified format
5. If no clear relationships exist, return empty relationships array

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
            "description": "clear description of relationship"
        }}
    ]
}}

Important:
- Use exact entity names from the entity list
- Create only relationships clearly supported by the text
- If unsure, use "RELATED_TO" as relationship type
- Return only valid JSON, no additional text or explanations

Return only the JSON object, no additional text.
""",
    
    "zh": """
您是一个专业的关系提取系统。分析给定文本中实体之间的关系。

说明：
1. 识别提供的实体之间的重要关系
2. 对于每个关系，请指定：
   - source: 源实体名称（与实体列表中的精确匹配）
   - target: 目标实体名称（与实体列表中的精确匹配）
   - relationship_type: 关系类型 [RELATED_TO, WORKS_FOR, LOCATED_IN, CREATED_BY]
   - confidence: 基于您的置信度的 0.0 到 1.0 的分数
   - description: 关系的清晰描述
3. 仅包含文本中明确提及或强烈暗示的关系
4. 仅返回指定格式的有效 JSON
5. 如果没有明确的关系，返回空的关系数组

要分析的实体：{entities}

要分析的文本：
{text}

预期的 JSON 格式：
{{
    "relationships": [
        {{
            "source": "实体名称",
            "target": "实体名称",
            "relationship_type": "RELATED_TO",
            "confidence": 0.8,
            "description": "关系描述"
        }}
    ]
}}

重要：
- 使用实体列表中的确切实体名称
- 如果不确定，使用 "RELATED_TO" 作为关系类型
- 仅返回有效 JSON，不包含额外文本或解释

仅返回 JSON 对象，不包含额外文本。
""",
    
    "ru": """
Вы экспертная система извлечения отношений. Проанализируйте отношения между сущностями.

Инструкции:
1. Найдите отношения между сущностями
2. Для каждого отношения укажите:
   - source: имя исходной сущности
   - target: имя целевой сущности  
   - relationship_type: тип отношения [RELATED_TO, WORKS_FOR, LOCATED_IN, CREATED_BY]
   - confidence: оценка 0.0-1.0
   - description: описание отношения
3. Возвращайте только JSON

Сущности: {entities}
Текст: {text}

Формат JSON:
{{
    "relationships": [
        {{
            "source": "сущность1",
            "target": "сущность2", 
            "relationship_type": "RELATED_TO",
            "confidence": 0.8,
            "description": "описание"
        }}
    ]
}}

Только JSON, без дополнительного текста.
"""
}

# Language detection patterns
LANGUAGE_DETECTION_PATTERNS = {
    "en": [
        r'\b(?:the|and|or|but|in|on|at|to|for|of|with|by|from|about|into|through|during|before|after|above|below)\b',
        r'\b(?:is|are|was|were|be|been|being|have|has|had|do|does|did|will|would|could|should|may|might|can)\b',
        r'\b(?:a|an|this|that|these|those|my|your|his|her|its|our|their)\b'
    ],
    "zh": [
        r'[\u4e00-\u9fff]',  # Chinese characters
        r'(?:的|是|在|有|和|与|或|但|而|如果|因为|所以|虽然|但是|然后|现在|以前|以后)',
        r'(?:我|你|他|她|它|我们|你们|他们|她们|它们)'
    ],
    "ru": [
        r'[а-яё]',  # Russian characters
        r'\b(?:и|в|на|с|по|для|от|до|из|к|у|о|об|при|про|за|над|под|перед|после|через|между|вокруг|внутри|снаружи)\b',
        r'\b(?:быть|был|была|были|было|есть|стать|стал|стала|стали|стало|являться|является|являлся|являлась|являлись)\b',
        r'\b(?:я|ты|он|она|оно|мы|вы|они|мой|твой|его|её|наш|ваш|их)\b'
    ]
}


def get_language_regex_patterns(language: str) -> Dict[str, List[str]]:
    """Get regex patterns for a specific language."""
    return LANGUAGE_REGEX_PATTERNS.get(language, LANGUAGE_REGEX_PATTERNS["en"])


def get_language_processing_settings(language: str) -> Dict[str, Any]:
    """Get processing settings for a specific language."""
    return LANGUAGE_PROCESSING_SETTINGS.get(language, LANGUAGE_PROCESSING_SETTINGS["en"])


def get_language_relationship_prompt(language: str) -> str:
    """Get relationship mapping prompt for a specific language."""
    return LANGUAGE_RELATIONSHIP_PROMPTS.get(language, LANGUAGE_RELATIONSHIP_PROMPTS["en"])


def get_language_detection_patterns(language: str) -> List[str]:
    """Get language detection patterns for a specific language."""
    return LANGUAGE_DETECTION_PATTERNS.get(language, LANGUAGE_DETECTION_PATTERNS["en"])


def detect_language_from_text(text: str) -> str:
    """Detect the primary language of the text using regex patterns."""
    if not text:
        return "en"
    
    # Count matches for each language
    language_scores = {}
    
    for lang, patterns in LANGUAGE_DETECTION_PATTERNS.items():
        score = 0
        for pattern in patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            score += matches
        language_scores[lang] = score
    
    # Return the language with the highest score
    if language_scores:
        return max(language_scores, key=language_scores.get)
    
    return "en"


def should_use_simplified_prompt(language: str) -> bool:
    """Check if simplified relationship prompt should be used for a language."""
    settings = get_language_processing_settings(language)
    return settings.get("relationship_prompt_simplified", False)


def get_entity_extraction_config(language: str) -> Dict[str, Any]:
    """Get comprehensive entity extraction configuration for a language."""
    patterns = get_language_regex_patterns(language)
    settings = get_language_processing_settings(language)
    
    return {
        "patterns": patterns,
        "settings": settings,
        "use_simplified_prompt": should_use_simplified_prompt(language),
        "relationship_prompt": get_language_relationship_prompt(language)
    }
