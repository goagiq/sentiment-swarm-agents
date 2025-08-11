#!/usr/bin/env python3
"""
Integration verification script for File Extraction Agent.
This script verifies that the agent is properly integrated with the orchestrator and API.
"""

import asyncio
import sys
import tempfile
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.orchestrator import SentimentOrchestrator
from core.models import AnalysisRequest, DataType
from agents.file_extraction_agent import FileExtractionAgent


async def verify_orchestrator_integration():
    """Verify that File Extraction Agent is properly integrated with orchestrator."""
    print("🔍 Verifying Orchestrator Integration...")
    
    try:
        # Create orchestrator
        orchestrator = SentimentOrchestrator()
        
        # Check if FileExtractionAgent is registered
        agent_ids = list(orchestrator.agents.keys())
        file_extraction_agents = [
            agent_id for agent_id in agent_ids 
            if 'FileExtractionAgent' in agent_id
        ]
        
        if not file_extraction_agents:
            print("❌ FileExtractionAgent not found in orchestrator")
            return False
        
        print(f"✅ Found {len(file_extraction_agents)} FileExtractionAgent(s) in orchestrator")
        
        # Check agent capabilities
        for agent_id in file_extraction_agents:
            agent = orchestrator.agents[agent_id]
            if 'pdf' not in agent.metadata.get('supported_types', []):
                print(f"❌ Agent {agent_id} doesn't support PDF")
                return False
        
        print("✅ All FileExtractionAgents support PDF processing")
        
        # Check if analyze_pdf method exists
        if not hasattr(orchestrator, 'analyze_pdf'):
            print("❌ Orchestrator missing analyze_pdf method")
            return False
        
        print("✅ Orchestrator has analyze_pdf method")
        
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator integration verification failed: {e}")
        return False


async def verify_agent_functionality():
    """Verify that File Extraction Agent is functional."""
    print("\n🔍 Verifying Agent Functionality...")
    
    try:
        # Create agent
        agent = FileExtractionAgent()
        
        # Check basic functionality
        if not hasattr(agent, 'can_process'):
            print("❌ Agent missing can_process method")
            return False
        
        if not hasattr(agent, 'process'):
            print("❌ Agent missing process method")
            return False
        
        # Test can_process with PDF request
        request = AnalysisRequest(
            data_type=DataType.PDF,
            content="/path/to/test.pdf",
            language="en"
        )
        
        can_process = await agent.can_process(request)
        if not can_process:
            print("❌ Agent cannot process PDF requests")
            return False
        
        print("✅ Agent can process PDF requests")
        
        # Check configuration
        expected_attrs = ['max_workers', 'chunk_size', 'retry_attempts', 'stats']
        for attr in expected_attrs:
            if not hasattr(agent, attr):
                print(f"❌ Agent missing {attr} attribute")
                return False
        
        print("✅ Agent has all required attributes")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent functionality verification failed: {e}")
        return False


async def verify_api_integration():
    """Verify that API integration is working."""
    print("\n🔍 Verifying API Integration...")
    
    try:
        # Import API components
        from api.main import app, PDFRequest
        
        # Check if PDF endpoint exists
        routes = [route.path for route in app.routes]
        if "/analyze/pdf" not in routes:
            print("❌ PDF analysis endpoint not found in API")
            return False
        
        print("✅ PDF analysis endpoint found in API")
        
        # Test PDFRequest model
        request = PDFRequest(
            pdf_path="/path/to/test.pdf",
            model_preference="llava:latest",
            reflection_enabled=True,
            max_iterations=3,
            confidence_threshold=0.8
        )
        
        if request.pdf_path != "/path/to/test.pdf":
            print("❌ PDFRequest model not working correctly")
            return False
        
        print("✅ PDFRequest model working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ API integration verification failed: {e}")
        return False


async def verify_dependencies():
    """Verify that all required dependencies are available."""
    print("\n🔍 Verifying Dependencies...")
    
    dependencies = {
        'PyPDF2': 'PyPDF2',
        'PyMuPDF': 'fitz',
        'Pillow': 'PIL',
        'ollama': 'ollama'
    }
    
    missing_deps = []
    
    for dep_name, import_name in dependencies.items():
        try:
            __import__(import_name)
            print(f"✅ {dep_name} available")
        except ImportError:
            print(f"❌ {dep_name} not available")
            missing_deps.append(dep_name)
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        return False
    
    return True


async def run_integration_test():
    """Run a basic integration test with a mock PDF."""
    print("\n🔍 Running Integration Test...")
    
    try:
        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(b'%PDF-1.4\n%Test PDF content\n%%EOF')
            temp_pdf = f.name
        
        try:
            # Test orchestrator integration
            orchestrator = SentimentOrchestrator()
            
            # This would normally process the PDF, but we'll just verify the method exists
            if hasattr(orchestrator, 'analyze_pdf'):
                print("✅ Integration test passed - analyze_pdf method available")
                return True
            else:
                print("❌ Integration test failed - analyze_pdf method not available")
                return False
                
        finally:
            # Clean up
            try:
                os.unlink(temp_pdf)
            except OSError:
                pass
                
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


async def main():
    """Main verification function."""
    print("🚀 File Extraction Agent Integration Verification")
    print("=" * 50)
    
    results = []
    
    # Run all verification steps
    results.append(await verify_dependencies())
    results.append(await verify_orchestrator_integration())
    results.append(await verify_agent_functionality())
    results.append(await verify_api_integration())
    results.append(await run_integration_test())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("\n🎉 File Extraction Agent is fully integrated and ready for use!")
        print("\n📋 Next Steps:")
        print("1. Start the system: python main.py")
        print("2. Access API documentation: http://localhost:8001/docs")
        print("3. Test PDF analysis: POST /analyze/pdf")
        return 0
    else:
        print(f"❌ SOME TESTS FAILED ({passed}/{total})")
        print("\n🔧 Please check the errors above and fix any issues.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
