#!/usr/bin/env python3
"""
Debug script to specifically test Russian PDF processing and identify issues.
This will help us understand why Russian PDF importing stopped working after Chinese optimizations.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.file_extraction_agent import FileExtractionAgent
from agents.knowledge_graph_agent import KnowledgeGraphAgent
from agents.entity_extraction_agent import EntityExtractionAgent
from core.models import AnalysisRequest, DataType
from config.language_specific_config import (
    detect_primary_language, 
    get_language_config,
    should_use_enhanced_extraction,
    get_language_patterns,
    get_language_dictionaries
)


async def debug_russian_pdf_processing():
    """Debug Russian PDF processing step by step."""
    
    print("🔍 DEBUGGING RUSSIAN PDF PROCESSING")
    print("=" * 80)
    
    pdf_path = "data/Russian_Oliver_Excerpt.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    try:
        # Step 1: Test File Extraction
        print("\n📄 Step 1: Testing File Extraction...")
        print("   ⏳ Initializing FileExtractionAgent...")
        file_agent = FileExtractionAgent()
        print("   ✅ FileExtractionAgent initialized")
        
        print("   ⏳ Creating analysis request...")
        pdf_request = AnalysisRequest(
            data_type=DataType.PDF,
            content=pdf_path,
            language="ru"
        )
        print("   ✅ Analysis request created")
        
        print("   ⏳ Processing PDF extraction (this may take a moment)...")
        extraction_result = await file_agent.process(pdf_request)
        print("   ✅ PDF extraction completed")
        
        if extraction_result.status != "completed":
            print(f"❌ File extraction failed: {extraction_result.metadata.get('error', 'Unknown error')}")
            return False
        
        text_content = extraction_result.extracted_text
        print(f"   📊 Content length: {len(text_content)} characters")
        print(f"   📄 Pages processed: {len(extraction_result.pages) if extraction_result.pages else 'Unknown'}")
        print(f"   ⏱️ Processing time: {extraction_result.processing_time:.2f}s")
        
        # Step 2: Test Language Detection
        print("\n🌍 Step 2: Testing Language Detection...")
        print("   ⏳ Detecting language from extracted text...")
        detected_language = detect_primary_language(text_content)
        print(f"   ✅ Language detection result: {detected_language}")
        print(f"   📋 Expected: ru, Detected: {detected_language}")
        
        # Step 3: Test Language Configuration
        print("\n⚙️ Step 3: Testing Language Configuration...")
        print("   ⏳ Checking Russian language configuration...")
        
        # Check if Russian config exists
        ru_config = get_language_config("ru")
        print(f"   ✅ Russian config found: {bool(ru_config)}")
        
        if ru_config:
            print(f"   📋 Enhanced extraction enabled: {ru_config.get('use_enhanced_extraction', False)}")
            print(f"   📋 Min entity length: {ru_config.get('min_entity_length', 'Not set')}")
            print(f"   📋 Max entity length: {ru_config.get('max_entity_length', 'Not set')}")
            print(f"   📋 Confidence threshold: {ru_config.get('confidence_threshold', 'Not set')}")
        
        # Check patterns
        ru_patterns = get_language_patterns("ru")
        print(f"   📋 Russian patterns found: {len(ru_patterns)} entity types")
        for entity_type, patterns in ru_patterns.items():
            print(f"      - {entity_type}: {len(patterns)} patterns")
        
        # Check dictionaries
        ru_dictionaries = get_language_dictionaries("ru")
        print(f"   📋 Russian dictionaries found: {len(ru_dictionaries)} entity types")
        for entity_type, entities in ru_dictionaries.items():
            print(f"      - {entity_type}: {len(entities)} entities")
        
        # Step 4: Test Entity Extraction
        print("\n🧠 Step 4: Testing Entity Extraction...")
        print("   ⏳ Initializing EntityExtractionAgent...")
        entity_agent = EntityExtractionAgent()
        print("   ✅ EntityExtractionAgent initialized")
        
        print("   ⏳ Testing basic entity extraction...")
        basic_result = await entity_agent.extract_entities(text_content)
        print(f"   ✅ Basic extraction completed: {basic_result.get('count', 0)} entities found")
        
        print("   ⏳ Testing multilingual entity extraction...")
        multilingual_result = await entity_agent.extract_entities_multilingual(text_content, "ru")
        print(f"   ✅ Multilingual extraction completed: {multilingual_result.get('count', 0)} entities found")
        
        print("   ⏳ Testing enhanced Russian entity extraction...")
        try:
            enhanced_result = await entity_agent._extract_russian_entities_enhanced(text_content)
            print(f"   ✅ Enhanced extraction completed: {enhanced_result.get('count', 0)} entities found")
            print(f"   📋 Extraction method: {enhanced_result.get('extraction_method', 'Unknown')}")
            
            if enhanced_result.get('entities'):
                print("   📋 Sample entities:")
                for i, entity in enumerate(enhanced_result['entities'][:5]):
                    print(f"      {i+1}. {entity.get('text', 'N/A')} ({entity.get('type', 'N/A')}) - {entity.get('confidence', 0):.2f}")
        except Exception as e:
            print(f"   ❌ Enhanced extraction failed: {e}")
        
        # Step 5: Test Knowledge Graph Processing
        print("\n🧠 Step 5: Testing Knowledge Graph Processing...")
        print("   ⏳ Initializing KnowledgeGraphAgent...")
        kg_agent = KnowledgeGraphAgent()
        print("   ✅ KnowledgeGraphAgent initialized")
        
        print("   ⏳ Creating knowledge graph request...")
        kg_request = AnalysisRequest(
            data_type=DataType.TEXT,
            content=text_content,
            language=detected_language
        )
        print("   ✅ Knowledge graph request created")
        
        print("   ⏳ Processing knowledge graph (this may take a moment)...")
        start_time = datetime.now()
        kg_result = await kg_agent.process(kg_request)
        processing_time = (datetime.now() - start_time).total_seconds()
        print(f"   ✅ Knowledge graph processing completed in {processing_time:.2f}s")
        
        if kg_result.status == "completed":
            print(f"   ✅ Knowledge graph processing successful")
            print(f"   📊 Processing time: {processing_time:.2f}s")
            print(f"   📋 Status: {kg_result.status}")
            
            # Check metadata for entity information
            if 'entities_extracted' in kg_result.metadata:
                print(f"   📊 Entities extracted: {kg_result.metadata['entities_extracted']}")
            
            if 'relationships_found' in kg_result.metadata:
                print(f"   🔗 Relationships found: {kg_result.metadata['relationships_found']}")
            
            # Check for Russian-specific content in extracted text
            if kg_result.extracted_text:
                russian_keywords = ['Путин', 'Москва', 'Россия', 'Газпром', 'Хрущёв']
                found_keywords = [kw for kw in russian_keywords if kw in kg_result.extracted_text]
                print(f"   🇷🇺 Russian keywords found: {len(found_keywords)}")
                if found_keywords:
                    print(f"   📋 Found keywords: {', '.join(found_keywords)}")
        else:
            print(f"   ❌ Knowledge graph processing failed: {kg_result.metadata.get('error', 'Unknown error')}")
        
        print("\n✅ Russian PDF processing debug completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Debug failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main debug function."""
    print("🚀 Starting Russian PDF Processing Debug...")
    print(f"📅 Debug started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = await debug_russian_pdf_processing()
    
    print(f"\n📊 Debug Summary:")
    print(f"   {'✅ PASSED' if success else '❌ FAILED'}")
    print(f"   📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    asyncio.run(main())
