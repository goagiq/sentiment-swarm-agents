"""
Hindi language configuration for enhanced processing.
Enhanced with comprehensive regex patterns and grammar optimization.
"""

from typing import Dict, List
from .base_config import BaseLanguageConfig, EntityPatterns, ProcessingSettings


class HindiConfig(BaseLanguageConfig):
    """Hindi language configuration with enhanced regex patterns and grammar support."""
    
    def __init__(self):
        super().__init__()
        self.language_code = "hi"
        self.language_name = "Hindi"
        self.entity_patterns = self.get_entity_patterns()
        self.processing_settings = self.get_processing_settings()
        self.relationship_templates = self.get_relationship_templates()
        self.detection_patterns = self.get_detection_patterns()
        self.grammar_patterns = self.get_grammar_patterns()
        self.advanced_patterns = self.get_advanced_patterns()
    
    def get_entity_patterns(self) -> EntityPatterns:
        """Get enhanced Hindi entity patterns."""
        return EntityPatterns(
            person=[
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
            organization=[
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
            location=[
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
            concept=[
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
        )
    
    def get_grammar_patterns(self) -> Dict[str, List[str]]:
        """Get comprehensive grammar patterns for Hindi."""
        return {
            "verb_forms": [
                r'[\u0900-\u097F]+(?:ना|नी|ने|ता|ती|ते|ता|ती|ते)',  # Verb endings
                r'[\u0900-\u097F]+(?:कर|हो|जा|आ|जा|दे|ले|पा|सक|चाह)',  # Common verb roots
                r'[\u0900-\u097F]+(?:रहा|रही|रहे|गया|गई|गए|किया|की|किए)',  # Past tense
            ],
            "postpositions": [
                r'\b(?:का|की|के|में|पर|से|तक|द्वारा|के लिए|के बारे में)\b',
                r'\b(?:के साथ|के बिना|के पास|के पीछे|के सामने|के नीचे|के ऊपर)\b',
            ],
            "conjunctions": [
                r'\b(?:और|या|लेकिन|मगर|परंतु|किंतु|तथा|एवं|अथवा|या फिर)\b',
                r'\b(?:क्योंकि|क्योंकि|इसलिए|ताकि|जब|जबकि|हालांकि|यद्यपि)\b',
            ],
            "question_words": [
                r'\b(?:क्या|कौन|कहाँ|कब|कैसे|क्यों|कितना|कितनी|कितने|कौन सा)\b',
                r'\b(?:क्या|क्या|क्या|क्या|क्या|क्या|क्या|क्या)\b',
            ],
            "numbers": [
                r'\b(?:शून्य|एक|दो|तीन|चार|पांच|छह|सात|आठ|नौ|दस)\b',
                r'\b(?:पहला|दूसरा|तीसरा|चौथा|पांचवां|छठा|सातवां|आठवां|नौवां|दसवां)\b',
            ],
            "honorifics": [
                r'\b(?:श्री|श्रीमती|डॉ\.|प्रोफेसर|मिस्टर|मिस|कुमारी|बाबू|साहब|जी)\b',
                r'\b(?:गुरु|स्वामी|महात्मा|पंडित|मौलवी|पादरी|सिस्टर|फादर)\b',
            ]
        }
    
    def get_advanced_patterns(self) -> Dict[str, List[str]]:
        """Get advanced Hindi patterns for specialized processing."""
        return {
            "sanskrit_terms": [
                r'(?:धर्म|कर्म|मोक्ष|माया|ब्रह्म|आत्मा|परमात्मा|मोक्ष)',
                r'(?:योग|ध्यान|प्राणायाम|आसन|मुद्रा|बंध|क्रिया)',
                r'(?:वेद|उपनिषद|पुराण|रामायण|महाभारत|गीता|मनुस्मृति)',
            ],
            "hindi_literature": [
                r'(?:कविता|कहानी|उपन्यास|नाटक|निबंध|लेख|रिपोर्ट)',
                r'(?:कबीर|तुलसी|सूर|मीरा|रहीम|रसखान|बिहारी|देव)',
            ],
            "indian_philosophy": [
                r'(?:वेदांत|सांख्य|योग|न्याय|वैशेषिक|मीमांसा|बौद्ध|जैन)',
                r'(?:शंकराचार्य|रामानुज|मध्व|निम्बार्क|वल्लभ|चैतन्य)',
            ],
            "indian_science": [
                r'(?:आयुर्वेद|सिद्ध|यूनानी|होम्योपैथी|नेचुरोपैथी)',
                r'(?:ज्योतिष|वास्तु|फेंगशुई|रेकी|प्राणिक चिकित्सा)',
            ],
            "indian_culture": [
                r'(?:भारतीय संस्कृति|हिंदू धर्म|सिख धर्म|जैन धर्म|बौद्ध धर्म)',
                r'(?:दीपावली|होली|रक्षाबंधन|करवा चौथ|रामनवमी|कृष्ण जन्माष्टमी)',
            ]
        }
    
    def get_processing_settings(self) -> ProcessingSettings:
        """Get Hindi-specific processing settings."""
        return ProcessingSettings(
            min_entity_length=2,  # Hindi can have shorter entities
            max_entity_length=25,  # Hindi entities are typically shorter
            confidence_threshold=0.75,
            use_enhanced_extraction=True,
            relationship_prompt_simplified=False,  # Hindi needs detailed prompts
            use_hierarchical_relationships=True,
            entity_clustering_enabled=True,
            fallback_strategies=[
                "hindi_patterns",
                "sanskrit_terms",
                "hierarchical",
                "semantic",
                "template"
            ]
        )
    
    def get_relationship_templates(self) -> Dict[str, List[str]]:
        """Get Hindi relationship templates."""
        return {
            "person_organization": [
                "{person} {organization} में काम करता है",
                "{person} {organization} का मैनेजर है",
                "{person} {organization} का प्रमुख है",
                "{person} {organization} का सदस्य है",
            ],
            "person_location": [
                "{person} {location} में रहता है",
                "{person} {location} से है",
                "{person} {location} में निवासी है",
                "{person} {location} में पैदा हुआ है",
            ],
            "organization_location": [
                "{organization} {location} में स्थित है",
                "{organization} का मुख्यालय {location} में है",
                "{organization} का शाखा {location} में है",
                "{organization} {location} में स्थापित किया गया है",
            ],
            "concept_related": [
                "{concept1} {concept2} से संबंधित है",
                "{concept1} {concept2} की ओर ले जाता है",
                "{concept1} {concept2} का हिस्सा है",
                "{concept1} {concept2} पर निर्भर है",
            ]
        }
    
    def get_detection_patterns(self) -> List[str]:
        """Get Hindi language detection patterns."""
        return [
            r'[\u0900-\u097F]+',  # Hindi Unicode range
            r'[\u0980-\u09FF]+',  # Bengali
            r'[\u0A00-\u0A7F]+',  # Gurmukhi
            r'[\u0A80-\u0AFF]+',  # Gujarati
            r'[\u0B00-\u0B7F]+',  # Oriya
            r'[\u0B80-\u0BFF]+',  # Tamil
            r'[\u0C00-\u0C7F]+',  # Telugu
            r'[\u0C80-\u0CFF]+',  # Kannada
            r'[\u0D00-\u0D7F]+',  # Malayalam
            r'\b(?:का|की|के|में|पर|से|तक|द्वारा|के लिए|के बारे में)\b',
            r'\b(?:और|या|लेकिन|मगर|परंतु|किंतु|तथा|एवं|अथवा|या फिर)\b',
            r'\b(?:है|हैं|था|थी|थे|था|थी|थे|हो|होती|होते|होता)\b',
            r'\b(?:हैं|है|था|थी|थे|था|थी|थे|हो|होती|होते|होता)\b',  # Standard Hindi
            r'\b(?:है|हैं|था|थी|थे|था|थी|थे|हो|होती|होते|होता)\b',  # Hindustani
            r'\b(?:है|हैं|था|थी|थे|था|थी|थे|हो|होती|होते|होता)\b',  # Urdu influence
        ]
