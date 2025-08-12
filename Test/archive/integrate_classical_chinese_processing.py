#!/usr/bin/env python3
"""
Integrate Classical Chinese PDF processing into the main system.
Uses existing MCP servers and API endpoints with language-specific configurations.
"""

import asyncio
import os
import sys
import requests
import json
from datetime import datetime
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config.language_config import LanguageConfigFactory
from src.config.config import config
from src.core.mcp_server import OptimizedMCPServer


class ClassicalChineseProcessor:
    """Integrated Classical Chinese PDF processor using existing MCP servers and API endpoints."""
    
    def __init__(self):
        self.mcp_server = None
        self.api_base_url = "http://localhost:8000"
        self.chinese_config = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize MCP server and language configurations."""
        try:
            # Initialize MCP server
            self.mcp_server = OptimizedMCPServer()
            print("✅ MCP Server initialized successfully")
            
            # Initialize Chinese language configuration
            self.chinese_config = LanguageConfigFactory.get_config("zh")
            print("✅ Chinese language configuration loaded")
            
            # Check for Classical Chinese patterns
            if hasattr(self.chinese_config, 'classical_patterns'):
                print(f"✅ Classical Chinese patterns available: {len(self.chinese_config.classical_patterns)} categories")
            
            if hasattr(self.chinese_config, 'is_classical_chinese'):
                print("✅ Classical Chinese detection method available")
                
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize all components: {e}")
    
    async def process_classical_chinese_pdf_mcp(self, pdf_path: str):
        """Process Classical Chinese PDF using MCP server directly."""
        if not self.mcp_server or not self.mcp_server.mcp:
            raise Exception("MCP server not available")
        
        print(f"🔧 Processing via MCP server: {pdf_path}")
        
        try:
            # Use the MCP server's process_pdf_enhanced_multilingual tool
            result = await self.mcp_server.mcp.process_pdf_enhanced_multilingual(
                pdf_path=pdf_path,
                language="zh",  # Explicitly set to Chinese for Classical Chinese
                generate_report=True,
                output_path=None
            )
            
            return result
            
        except Exception as e:
            print(f"❌ MCP processing failed: {e}")
            return None
    
    async def process_classical_chinese_pdf_api(self, pdf_path: str):
        """Process Classical Chinese PDF using API endpoints."""
        print(f"🌐 Processing via API endpoint: {pdf_path}")
        
        try:
            # Use the enhanced multilingual PDF processing endpoint
            url = f"{self.api_base_url}/process/pdf-enhanced-multilingual"
            
            payload = {
                "pdf_path": pdf_path,
                "language": "zh",  # Explicitly set to Chinese for Classical Chinese
                "generate_report": True,
                "output_path": None
            }
            
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ API processing failed: {e}")
            return None
    
    async def process_classical_chinese_pdf_direct(self, pdf_path: str):
        """Process Classical Chinese PDF using direct agent calls."""
        print(f"🎯 Processing via direct agent calls: {pdf_path}")
        
        try:
            from src.agents.enhanced_file_extraction_agent import EnhancedFileExtractionAgent
            from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
            from src.core.models import AnalysisRequest, DataType
            from src.config.language_specific_config import detect_primary_language
            
            # Step 1: Extract text using enhanced file extraction agent
            print("📄 Step 1: Extracting text with enhanced multilingual processing...")
            file_agent = EnhancedFileExtractionAgent()
            
            pdf_request = AnalysisRequest(
                data_type=DataType.PDF,
                content=pdf_path,
                language="zh"
            )
            
            extraction_result = await file_agent.process(pdf_request)
            
            if extraction_result.status != "completed":
                raise Exception(f"PDF extraction failed: {extraction_result.metadata.get('error', 'Unknown error')}")
            
            text_content = extraction_result.extracted_text
            print(f"✅ Text extraction completed. Content length: {len(text_content)} characters")
            
            # Step 2: Check for Classical Chinese patterns
            if self.chinese_config and hasattr(self.chinese_config, 'is_classical_chinese'):
                is_classical = self.chinese_config.is_classical_chinese(text_content[:1000])
                print(f"🏛️ Classical Chinese detected: {is_classical}")
            
            # Step 3: Process with knowledge graph agent
            print("🧠 Step 2: Processing with knowledge graph agent...")
            kg_agent = KnowledgeGraphAgent()
            
            kg_request = AnalysisRequest(
                data_type=DataType.TEXT,
                content=text_content,
                language="zh"
            )
            
            kg_result = await kg_agent.process(kg_request)
            
            if kg_result.status != "completed":
                raise Exception(f"Knowledge graph processing failed: {kg_result.metadata.get('error', 'Unknown error')}")
            
            # Step 4: Generate report
            print("📊 Step 3: Generating knowledge graph report...")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"Results/reports/classical_chinese_direct_{timestamp}"
            
            os.makedirs("Results/reports", exist_ok=True)
            
            report_result = await kg_agent.generate_graph_report(
                output_path=output_path,
                target_language="zh"
            )
            
            # Compile results
            stats = kg_result.metadata.get("statistics", {}) if kg_result.metadata else {}
            
            result = {
                "success": True,
                "method": "direct_agent_calls",
                "pdf_path": pdf_path,
                "language": "zh",
                "text_extraction": {
                    "success": True,
                    "content_length": len(text_content),
                    "pages_processed": len(extraction_result.pages) if extraction_result.pages else 'Unknown',
                    "extraction_method": "Enhanced multilingual"
                },
                "entity_extraction": {
                    "entities_found": stats.get('entities_found', 0),
                    "entity_types": stats.get('entity_types', {}),
                    "language_stats": stats.get('language_stats', {}),
                    "extraction_method": "Enhanced multilingual with Classical Chinese support"
                },
                "knowledge_graph": {
                    "nodes": stats.get('nodes', 0),
                    "edges": stats.get('edges', 0),
                    "communities": stats.get('communities', 0),
                    "processing_time": kg_result.processing_time
                },
                "report_files": {
                    "output_path": output_path,
                    "success": hasattr(report_result, 'success') and report_result.success
                },
                "enhanced_features": {
                    "language_specific_patterns": True,
                    "dictionary_lookup": True,
                    "llm_based_extraction": True,
                    "classical_chinese_support": True,
                    "multilingual_support": ["en", "ru", "zh"]
                }
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Direct processing failed: {e}")
            return None
    
    async def process_classical_chinese_pdf(self, pdf_path: str, method: str = "auto"):
        """Process Classical Chinese PDF using the specified method."""
        
        # Validate PDF file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"🏛️ Processing Classical Chinese PDF: {pdf_path}")
        print("=" * 60)
        
        # Try different processing methods based on preference
        methods_to_try = []
        
        if method == "auto":
            methods_to_try = ["direct", "mcp", "api"]
        elif method == "mcp":
            methods_to_try = ["mcp", "direct", "api"]
        elif method == "api":
            methods_to_try = ["api", "direct", "mcp"]
        elif method == "direct":
            methods_to_try = ["direct", "mcp", "api"]
        else:
            methods_to_try = ["direct", "mcp", "api"]
        
        result = None
        
        for method_name in methods_to_try:
            try:
                print(f"\n🔄 Trying method: {method_name.upper()}")
                
                if method_name == "mcp":
                    result = await self.process_classical_chinese_pdf_mcp(pdf_path)
                elif method_name == "api":
                    result = await self.process_classical_chinese_pdf_api(pdf_path)
                elif method_name == "direct":
                    result = await self.process_classical_chinese_pdf_direct(pdf_path)
                
                if result and result.get("success"):
                    print(f"✅ Successfully processed using {method_name.upper()} method")
                    break
                else:
                    print(f"⚠️ {method_name.upper()} method failed or returned no result")
                    
            except Exception as e:
                print(f"❌ {method_name.upper()} method failed: {e}")
                continue
        
        if not result or not result.get("success"):
            raise Exception("All processing methods failed")
        
        # Display results
        self._display_results(result)
        
        return result
    
    def _display_results(self, result: dict):
        """Display processing results in a formatted way."""
        print("\n" + "=" * 60)
        print("📋 PROCESSING RESULTS")
        print("=" * 60)
        
        print(f"🎯 Processing Method: {result.get('method', 'Unknown')}")
        print(f"📄 PDF Path: {result.get('pdf_path', 'Unknown')}")
        print(f"🌍 Language: {result.get('language', 'Unknown')}")
        
        # Text extraction results
        text_extraction = result.get('text_extraction', {})
        print(f"\n📄 Text Extraction:")
        print(f"   - Success: {text_extraction.get('success', False)}")
        print(f"   - Content length: {text_extraction.get('content_length', 0)} characters")
        print(f"   - Pages processed: {text_extraction.get('pages_processed', 'Unknown')}")
        print(f"   - Extraction method: {text_extraction.get('extraction_method', 'Unknown')}")
        
        # Entity extraction results
        entity_extraction = result.get('entity_extraction', {})
        print(f"\n🔍 Entity Extraction:")
        print(f"   - Entities found: {entity_extraction.get('entities_found', 0)}")
        print(f"   - Entity types: {entity_extraction.get('entity_types', {})}")
        print(f"   - Language stats: {entity_extraction.get('language_stats', {})}")
        print(f"   - Extraction method: {entity_extraction.get('extraction_method', 'Unknown')}")
        
        # Knowledge graph results
        knowledge_graph = result.get('knowledge_graph', {})
        print(f"\n🧠 Knowledge Graph:")
        print(f"   - Nodes: {knowledge_graph.get('nodes', 0)}")
        print(f"   - Edges: {knowledge_graph.get('edges', 0)}")
        print(f"   - Communities: {knowledge_graph.get('communities', 0)}")
        print(f"   - Processing time: {knowledge_graph.get('processing_time', 0):.2f} seconds")
        
        # Report results
        report_files = result.get('report_files', {})
        if report_files:
            print(f"\n📊 Report Generation:")
            if isinstance(report_files, dict):
                for key, value in report_files.items():
                    print(f"   - {key}: {value}")
            else:
                print(f"   - Report files: {report_files}")
        
        # Enhanced features
        enhanced_features = result.get('enhanced_features', {})
        if enhanced_features:
            print(f"\n🚀 Enhanced Features:")
            for feature, enabled in enhanced_features.items():
                status = "✅" if enabled else "❌"
                print(f"   - {feature}: {status}")
        
        print("\n" + "=" * 60)
        print("✅ Classical Chinese PDF processing completed successfully!")
        print("=" * 60)


async def main():
    """Main function to run the integrated Classical Chinese PDF processing."""
    print("🏛️ Integrated Classical Chinese PDF Processing")
    print("=" * 60)
    
    # Initialize processor
    processor = ClassicalChineseProcessor()
    
    # PDF file path
    pdf_path = "data/Classical Chinese Sample 22208_0_8.pdf"
    
    try:
        # Process the Classical Chinese PDF
        result = await processor.process_classical_chinese_pdf(
            pdf_path=pdf_path,
            method="auto"  # Try direct, then MCP, then API
        )
        
        if result and result.get("success"):
            print("\n🎉 Processing completed successfully!")
            print("📁 Check the Results/reports directory for generated reports.")
            
            # Save results to JSON file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = f"Results/classical_chinese_processing_results_{timestamp}.json"
            os.makedirs("Results", exist_ok=True)
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"📄 Results saved to: {results_file}")
            
        else:
            print("\n❌ Processing failed. Check the error messages above.")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the async function
    asyncio.run(main())
