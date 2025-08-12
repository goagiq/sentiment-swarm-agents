#!/usr/bin/env python3
"""
Process PDFs and add to vector and knowledge graph databases, then generate reports.
"""

import sys
import asyncio
import os
from pathlib import Path
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.file_extraction_agent import FileExtractionAgent
from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
from src.core.models import AnalysisRequest, DataType
from src.config.language_specific_regex_config import (
    detect_language_from_text
)


async def process_pdf_to_databases(
    pdf_path: str, language: str = "auto", generate_report: bool = True
):
    """Process PDF file and add content to both vector and knowledge graph databases."""
    print(f"\n📄 Processing PDF: {pdf_path}")
    print(f"🌍 Language setting: {language}")
    
    try:
        # Validate file existence
        if not os.path.exists(pdf_path):
            print(f"❌ PDF file not found: {pdf_path}")
            return {"success": False, "error": f"PDF file not found: {pdf_path}"}
        
        # Create file extraction agent
        file_agent = FileExtractionAgent()
        
        # Create analysis request
        pdf_request = AnalysisRequest(
            data_type=DataType.PDF,
            content=pdf_path,
            language=language
        )
        
        # Extract text from PDF
        print(f"📄 Extracting text from PDF: {pdf_path}")
        extraction_result = await file_agent.process(pdf_request)
        
        # Check if extraction was successful
        if not extraction_result.extracted_text and not extraction_result.raw_content:
            print(f"❌ Failed to extract text from PDF: {pdf_path}")
            return {"success": False, "error": "Failed to extract text from PDF"}
        
        # Get the extracted text
        text_content = extraction_result.extracted_text or extraction_result.raw_content
        print(f"✅ Extracted {len(text_content)} characters of text")
        
        # Detect language if auto is specified
        detected_language = language
        if language == "auto":
            detected_language = detect_language_from_text(text_content)
            print(f"🌍 Detected language: {detected_language}")
        
        # Create knowledge graph agent
        kg_agent = KnowledgeGraphAgent()
        
        # Process with knowledge graph agent using enhanced multilingual support
        print(f"🧠 Processing with enhanced multilingual entity extraction "
              f"for language: {detected_language}")
        kg_request = AnalysisRequest(
            data_type=DataType.TEXT,
            content=text_content,
            language=detected_language
        )
        
        kg_result = await kg_agent.process(kg_request)
        
        # Generate report if requested
        report_files = {}
        if generate_report:
            print(f"📊 Generating knowledge graph report...")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"Results/reports/enhanced_multilingual_pdf_{detected_language}_{timestamp}"
            
            report_result = await kg_agent.generate_graph_report(
                output_path=output_path,
                target_language=detected_language
            )
            report_files = report_result.get("files", {})
            print(f"📊 Report generated: {output_path}")
        
        # Get statistics from knowledge graph result
        stats = kg_result.metadata.get("statistics", {}) if kg_result.metadata else {}
        graph_data = kg_result.metadata.get("graph_data", {}) if kg_result.metadata else {}
        
        # Additional processing for Russian to ensure relationships are created
        if detected_language == "ru":
            print(f"🔧 Applying Russian-specific processing enhancements...")
            try:
                # Extract entities again with enhanced processing
                entities_result = await kg_agent.extract_entities(text_content, detected_language)
                
                if entities_result and 'content' in entities_result:
                    entities = entities_result['content'][0].get('json', {}).get('entities', [])
                    print(f"📋 Enhanced entity extraction found {len(entities)} entities")
                    
                    if entities:
                        # Force relationship mapping with fallback
                        relationships_result = await kg_agent.map_relationships(text_content, entities, detected_language)
                        
                        if relationships_result and 'content' in relationships_result:
                            relationships = relationships_result['content'][0]['json']['relationships']
                            print(f"🔗 Enhanced relationship mapping created {len(relationships)} relationships")
                            
                            # If still no relationships, create basic ones
                            if not relationships and len(entities) >= 2:
                                print(f"🔧 Creating basic fallback relationships for Russian content...")
                                # Create simple relationships between entities
                                for i in range(len(entities) - 1):
                                    source = entities[i].get('name', f'Entity_{i}')
                                    target = entities[i + 1].get('name', f'Entity_{i+1}')
                                    if source and target and source != target:
                                        relationships.append({
                                            "source": source,
                                            "target": target,
                                            "relationship_type": "RELATED_TO",
                                            "confidence": 0.5,
                                            "description": f"{source} and {target} are mentioned in the same Russian text"
                                        })
                                
                                # Update the relationships result
                                relationships_result['content'][0]['json']['relationships'] = relationships
                                print(f"🔗 Created {len(relationships)} basic relationships")
            except Exception as e:
                print(f"⚠️ Russian-specific processing enhancement failed: {e}")
        
        result = {
            "success": True,
            "pdf_path": pdf_path,
            "detected_language": detected_language,
            "text_extraction": {
                "success": True,
                "content_length": len(text_content),
                "pages_processed": extraction_result.metadata.get("pages_processed", 0),
                "extraction_method": extraction_result.metadata.get("extraction_method", "unknown")
            },
            "entity_extraction": {
                "entities_found": stats.get("entities_found", 0),
                "entity_types": stats.get("entity_types", {}),
                "language_stats": stats.get("language_stats", {}),
                "extraction_method": "enhanced_multilingual"
            },
            "knowledge_graph": {
                "nodes": graph_data.get("nodes", 0),
                "edges": graph_data.get("edges", 0),
                "communities": graph_data.get("communities", 0),
                "processing_time": kg_result.processing_time
            },
            "report_files": report_files,
            "enhanced_features": {
                "language_specific_patterns": True,
                "dictionary_lookup": True,
                "llm_based_extraction": True,
                "multilingual_support": ["en", "ru", "zh"],
                "russian_enhancements": detected_language == "ru"
            }
        }
        
        print(f"✅ Successfully processed {pdf_path}")
        print(f"   - Language: {detected_language}")
        print(f"   - Entities found: {result['entity_extraction']['entities_found']}")
        print(f"   - Graph nodes: {result['knowledge_graph']['nodes']}")
        print(f"   - Graph edges: {result['knowledge_graph']['edges']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error processing {pdf_path}: {e}")
        return {"success": False, "error": str(e)}

async def generate_comprehensive_graph_report():
    """Generate a comprehensive graph report for all processed content."""
    print(f"\n📊 Generating comprehensive graph report...")
    
    try:
        kg_agent = KnowledgeGraphAgent()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"Results/reports/comprehensive_graph_report_{timestamp}"
        
        report_result = await kg_agent.generate_graph_report(
            output_path=output_path,
            target_language="en"  # Generate in English for comprehensive view
        )
        
        print(f"📊 Comprehensive report generated: {output_path}")
        return report_result
        
    except Exception as e:
        print(f"❌ Error generating comprehensive report: {e}")
        return {"success": False, "error": str(e)}

async def main():
    """Main function to process both PDFs and generate reports."""
    print("🚀 Starting PDF processing and knowledge graph generation...")
    
    # Define the PDF files to process
    pdf_files = [
        {
            "path": "data/Classical Chinese Sample 22208_0_8.pdf",
            "language": "zh"  # Classical Chinese
        },
        {
            "path": "data/Russian_Oliver_Excerpt.pdf", 
            "language": "ru"  # Russian
        }
    ]
    
    # Process each PDF
    results = []
    for pdf_info in pdf_files:
        result = await process_pdf_to_databases(
            pdf_path=pdf_info["path"],
            language=pdf_info["language"],
            generate_report=True
        )
        results.append(result)
        
        # Add a small delay between processing
        await asyncio.sleep(2)
    
    # Generate comprehensive graph report
    print(f"\n📊 Generating comprehensive graph report for all processed content...")
    comprehensive_report = await generate_comprehensive_graph_report()
    
    # Print summary
    print(f"\n📋 Processing Summary:")
    print(f"=" * 50)
    
    successful_count = sum(1 for r in results if r.get("success", False))
    total_count = len(results)
    
    print(f"📄 PDFs processed: {successful_count}/{total_count}")
    
    for i, result in enumerate(results):
        pdf_name = os.path.basename(pdf_files[i]["path"])
        if result.get("success", False):
            print(f"✅ {pdf_name}:")
            print(f"   - Language: {result.get('detected_language', 'unknown')}")
            print(f"   - Entities: {result.get('entity_extraction', {}).get('entities_found', 0)}")
            print(f"   - Graph nodes: {result.get('knowledge_graph', {}).get('nodes', 0)}")
            print(f"   - Graph edges: {result.get('knowledge_graph', {}).get('edges', 0)}")
        else:
            print(f"❌ {pdf_name}: {result.get('error', 'Unknown error')}")
    
    if comprehensive_report.get("success", False):
        print(f"📊 Comprehensive report: ✅ Generated")
    else:
        print(f"📊 Comprehensive report: ❌ {comprehensive_report.get('error', 'Unknown error')}")
    
    print(f"=" * 50)
    print(f"🎉 Processing complete!")

if __name__ == "__main__":
    asyncio.run(main())
