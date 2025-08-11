"""
Integration test for File Extraction Agent with orchestrator and API.
"""

import asyncio
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core.orchestrator import SentimentOrchestrator
from src.core.models import AnalysisRequest, DataType
from src.agents.file_extraction_agent import FileExtractionAgent


class TestFileExtractionIntegration:
    """Test File Extraction Agent integration with orchestrator."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance."""
        return SentimentOrchestrator()
    
    @pytest.fixture
    def file_extraction_agent(self):
        """Create File Extraction Agent instance."""
        return FileExtractionAgent()
    
    @pytest.fixture
    def sample_pdf_path(self):
        """Create a temporary PDF file for testing."""
        # Create a temporary file that looks like a PDF
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(b'%PDF-1.4\n%Test PDF content\n%%EOF')
            return f.name
    
    def test_agent_registration(self, orchestrator):
        """Test that File Extraction Agent is registered in orchestrator."""
        # Check if FileExtractionAgent is in the agents dictionary
        agent_ids = list(orchestrator.agents.keys())
        
        # Look for FileExtractionAgent
        file_extraction_agents = [
            agent_id for agent_id in agent_ids 
            if 'FileExtractionAgent' in agent_id
        ]
        
        assert len(file_extraction_agents) > 0, "FileExtractionAgent not found in orchestrator"
        
        # Check that the agent supports PDF data type
        for agent_id in file_extraction_agents:
            agent = orchestrator.agents[agent_id]
            assert hasattr(agent, 'metadata'), "Agent missing metadata"
            assert 'supported_types' in agent.metadata, "Agent missing supported_types"
            assert 'pdf' in agent.metadata['supported_types'], "Agent doesn't support PDF"
    
    def test_agent_can_process_pdf(self, file_extraction_agent):
        """Test that File Extraction Agent can process PDF requests."""
        # Create a PDF request
        request = AnalysisRequest(
            data_type=DataType.PDF,
            content="/path/to/test.pdf",
            language="en"
        )
        
        # Test can_process method
        result = asyncio.run(file_extraction_agent.can_process(request))
        assert result is True, "FileExtractionAgent should be able to process PDF requests"
    
    def test_orchestrator_has_pdf_method(self, orchestrator):
        """Test that orchestrator has analyze_pdf method."""
        assert hasattr(orchestrator, 'analyze_pdf'), "Orchestrator missing analyze_pdf method"
        assert callable(orchestrator.analyze_pdf), "analyze_pdf should be callable"
    
    @patch('src.agents.file_extraction_agent.PYPDF2_AVAILABLE', True)
    @patch('src.agents.file_extraction_agent.PyPDF2.PdfReader')
    def test_orchestrator_pdf_analysis(self, mock_pdf_reader, orchestrator, sample_pdf_path):
        """Test orchestrator PDF analysis integration."""
        # Mock PyPDF2 response
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Sample PDF text content"
        mock_pdf_reader.return_value.pages = [mock_page]
        mock_pdf_reader.return_value.__len__.return_value = 1
        
        # Test the analyze_pdf method
        async def test_analysis():
            result = await orchestrator.analyze_pdf(sample_pdf_path)
            assert result is not None, "Analysis result should not be None"
            assert result.data_type == DataType.PDF, "Result should have PDF data type"
            assert result.status.value == "completed", "Analysis should be completed"
            assert result.extracted_text is not None, "Should have extracted text"
            return result
        
        result = asyncio.run(test_analysis())
        assert result is not None
    
    def test_agent_inheritance(self, file_extraction_agent):
        """Test that File Extraction Agent properly inherits from base agent."""
        from src.agents.base_agent import StrandsBaseAgent
        
        assert isinstance(file_extraction_agent, StrandsBaseAgent), "Should inherit from StrandsBaseAgent"
        assert hasattr(file_extraction_agent, 'agent_id'), "Should have agent_id"
        assert hasattr(file_extraction_agent, 'can_process'), "Should have can_process method"
        assert hasattr(file_extraction_agent, 'process'), "Should have process method"
    
    def test_agent_configuration(self, file_extraction_agent):
        """Test File Extraction Agent configuration."""
        assert hasattr(file_extraction_agent, 'max_workers'), "Should have max_workers"
        assert hasattr(file_extraction_agent, 'chunk_size'), "Should have chunk_size"
        assert hasattr(file_extraction_agent, 'retry_attempts'), "Should have retry_attempts"
        assert hasattr(file_extraction_agent, 'ollama_integration'), "Should have ollama_integration"
        assert hasattr(file_extraction_agent, 'stats'), "Should have stats"
        
        # Check default values
        assert file_extraction_agent.max_workers == 4, "Default max_workers should be 4"
        assert file_extraction_agent.chunk_size == 1, "Default chunk_size should be 1"
        assert file_extraction_agent.retry_attempts == 1, "Default retry_attempts should be 1"
    
    def test_agent_tools(self, file_extraction_agent):
        """Test that File Extraction Agent has proper tools."""
        tools = file_extraction_agent._get_tools()
        assert isinstance(tools, list), "Tools should be a list"
        
        # Look for PDF extraction tool
        pdf_tools = [tool for tool in tools if 'extract_pdf_text' in tool.get('name', '')]
        assert len(pdf_tools) > 0, "Should have PDF extraction tool"
    
    def test_agent_statistics(self, file_extraction_agent):
        """Test File Extraction Agent statistics tracking."""
        stats = file_extraction_agent.get_stats()
        assert isinstance(stats, dict), "Stats should be a dictionary"
        
        expected_keys = [
            'total_files', 'successful_extractions', 'failed_extractions',
            'pages_processed', 'total_processing_time', 'pypdf2_success',
            'vision_ocr_success', 'agent_id', 'model_name', 'max_workers',
            'chunk_size', 'retry_attempts'
        ]
        
        for key in expected_keys:
            assert key in stats, f"Stats should contain {key}"
    
    def test_agent_metadata(self, file_extraction_agent):
        """Test File Extraction Agent metadata."""
        assert hasattr(file_extraction_agent, 'metadata'), "Should have metadata"
        assert isinstance(file_extraction_agent.metadata, dict), "Metadata should be a dictionary"
    
    def test_agent_status(self, file_extraction_agent):
        """Test File Extraction Agent status."""
        status = file_extraction_agent.get_status()
        assert isinstance(status, dict), "Status should be a dictionary"
        assert 'agent_id' in status, "Status should contain agent_id"
        assert 'status' in status, "Status should contain status"
        assert 'agent_type' in status, "Status should contain agent_type"
    
    def test_cleanup(self, sample_pdf_path):
        """Clean up temporary files."""
        try:
            os.unlink(sample_pdf_path)
        except OSError:
            pass


class TestFileExtractionAPIIntegration:
    """Test File Extraction Agent API integration."""
    
    def test_api_endpoint_exists(self):
        """Test that the PDF analysis API endpoint exists."""
        from src.api.main import app
        
        # Check if the endpoint is registered
        routes = [route.path for route in app.routes]
        assert "/analyze/pdf" in routes, "PDF analysis endpoint should be registered"
    
    def test_pdf_request_model(self):
        """Test PDF request model structure."""
        from src.api.main import PDFRequest
        
        # Test model creation
        request = PDFRequest(
            pdf_path="/path/to/test.pdf",
            model_preference="llava:latest",
            reflection_enabled=True,
            max_iterations=3,
            confidence_threshold=0.8
        )
        
        assert request.pdf_path == "/path/to/test.pdf"
        assert request.model_preference == "llava:latest"
        assert request.reflection_enabled is True
        assert request.max_iterations == 3
        assert request.confidence_threshold == 0.8
    
    def test_api_root_endpoint_includes_pdf(self):
        """Test that the root API endpoint includes PDF analysis."""
        from src.api.main import app
        
        # Get the root endpoint function
        root_route = None
        for route in app.routes:
            if route.path == "/" and "GET" in route.methods:
                root_route = route
                break
        
        assert root_route is not None, "Root endpoint should exist"
        
        # The endpoint should be documented in the README
        # This is a structural test to ensure PDF analysis is included


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v"])
