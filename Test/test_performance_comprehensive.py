#!/usr/bin/env python3
"""
Comprehensive Performance Test for Consolidated MCP Server
Tests all 6 core functions across all 4 categories with actual data
"""

import os
import sys
import time
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.mcp_config import get_consolidated_mcp_config, ProcessingCategory
from mcp.consolidated_mcp_server import ConsolidatedMCPServer

class PerformanceTester:
    def __init__(self):
        self.config = get_consolidated_mcp_config()
        self.server = ConsolidatedMCPServer(self.config)
        self.results = {}
        self.test_data_dir = Path("Test/test_data")
        self.test_data_dir.mkdir(exist_ok=True)
        
    def log_test(self, category: str, function: str, status: str, duration: float, details: str = ""):
        """Log test results"""
        if category not in self.results:
            self.results[category] = {}
        if function not in self.results[category]:
            self.results[category][function] = []
            
        self.results[category][function].append({
            "status": status,
            "duration": duration,
            "details": details,
            "timestamp": time.time()
        })
        
    def create_test_pdf(self) -> str:
        """Create a simple test PDF file"""
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            pdf_path = self.test_data_dir / "test_document.pdf"
            c = canvas.Canvas(str(pdf_path), pagesize=letter)
            c.drawString(100, 750, "Test PDF Document")
            c.drawString(100, 700, "This is a test document for performance testing.")
            c.drawString(100, 650, "It contains multiple lines of text in English.")
            c.drawString(100, 600, "Testing PDF processing capabilities.")
            c.save()
            return str(pdf_path)
        except ImportError:
            # Fallback: create a text file with .pdf extension
            pdf_path = self.test_data_dir / "test_document.pdf"
            with open(pdf_path, 'w', encoding='utf-8') as f:
                f.write("Test PDF Document\n")
                f.write("This is a test document for performance testing.\n")
                f.write("It contains multiple lines of text in English.\n")
                f.write("Testing PDF processing capabilities.\n")
            return str(pdf_path)
    
    def create_test_audio(self) -> str:
        """Create a simple test audio file"""
        try:
            import numpy as np
            from scipy.io import wavfile
            
            audio_path = self.test_data_dir / "test_audio.wav"
            # Create a simple sine wave
            sample_rate = 44100
            duration = 2  # seconds
            frequency = 440  # Hz
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
            audio_data = (audio_data * 32767).astype(np.int16)
            wavfile.write(str(audio_path), sample_rate, audio_data)
            return str(audio_path)
        except ImportError:
            # Fallback: create an empty file
            audio_path = self.test_data_dir / "test_audio.wav"
            audio_path.touch()
            return str(audio_path)
    
    def create_test_video(self) -> str:
        """Create a simple test video file"""
        try:
            import cv2
            import numpy as np
            
            video_path = self.test_data_dir / "test_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, 20.0, (640,480))
            
            for i in range(50):  # 2.5 seconds at 20fps
                frame = np.zeros((480,640,3), dtype=np.uint8)
                cv2.putText(frame, f'Test Frame {i}', (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                out.write(frame)
            out.release()
            return str(video_path)
        except ImportError:
            # Fallback: create an empty file
            video_path = self.test_data_dir / "test_video.mp4"
            video_path.touch()
            return str(video_path)
    
    def create_test_website_content(self) -> str:
        """Create test website content"""
        html_path = self.test_data_dir / "test_website.html"
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Website</title>
        </head>
        <body>
            <h1>Test Website Content</h1>
            <p>This is a test website for performance testing.</p>
            <p>It contains multiple paragraphs of text in English.</p>
            <p>Testing website processing capabilities.</p>
        </body>
        </html>
        """
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return str(html_path)
    
    def test_pdf_functions(self):
        """Test all 6 functions for PDF category"""
        print("\n🔍 Testing PDF Processing Functions...")
        
        pdf_path = self.create_test_pdf()
        
        # Test 1: Extract text
        start_time = time.time()
        try:
            if hasattr(self.server, 'pdf_server') and self.server.pdf_server:
                result = self.server.pdf_server.extract_text(pdf_path)
                duration = time.time() - start_time
                self.log_test("PDF", "extract_text", "PASS", duration, f"Extracted {len(result)} characters")
                print(f"  ✅ Extract text: {duration:.2f}s")
            else:
                self.log_test("PDF", "extract_text", "SKIP", 0, "PDF server not enabled")
                print("  ⏭️  Extract text: SKIPPED (server not enabled)")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("PDF", "extract_text", "FAIL", duration, str(e))
            print(f"  ❌ Extract text: {duration:.2f}s - {e}")
        
        # Test 2: Convert to image
        start_time = time.time()
        try:
            if hasattr(self.server, 'pdf_server') and self.server.pdf_server:
                result = self.server.pdf_server.convert_to_image(pdf_path)
                duration = time.time() - start_time
                self.log_test("PDF", "convert_to_image", "PASS", duration, f"Converted to {len(result)} images")
                print(f"  ✅ Convert to image: {duration:.2f}s")
            else:
                self.log_test("PDF", "convert_to_image", "SKIP", 0, "PDF server not enabled")
                print("  ⏭️  Convert to image: SKIPPED (server not enabled)")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("PDF", "convert_to_image", "FAIL", duration, str(e))
            print(f"  ❌ Convert to image: {duration:.2f}s - {e}")
        
        # Test 3: Summarization
        start_time = time.time()
        try:
            if hasattr(self.server, 'pdf_server') and self.server.pdf_server:
                result = self.server.pdf_server.summarize(pdf_path)
                duration = time.time() - start_time
                self.log_test("PDF", "summarize", "PASS", duration, f"Summary length: {len(result)}")
                print(f"  ✅ Summarize: {duration:.2f}s")
            else:
                self.log_test("PDF", "summarize", "SKIP", 0, "PDF server not enabled")
                print("  ⏭️  Summarize: SKIPPED (server not enabled)")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("PDF", "summarize", "FAIL", duration, str(e))
            print(f"  ❌ Summarize: {duration:.2f}s - {e}")
        
        # Test 4: Translation
        start_time = time.time()
        try:
            if hasattr(self.server, 'pdf_server') and self.server.pdf_server:
                result = self.server.pdf_server.translate(pdf_path, "en")
                duration = time.time() - start_time
                self.log_test("PDF", "translate", "PASS", duration, f"Translated to {len(result)} characters")
                print(f"  ✅ Translate: {duration:.2f}s")
            else:
                self.log_test("PDF", "translate", "SKIP", 0, "PDF server not enabled")
                print("  ⏭️  Translate: SKIPPED (server not enabled)")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("PDF", "translate", "FAIL", duration, str(e))
            print(f"  ❌ Translate: {duration:.2f}s - {e}")
        
        # Test 5: Vector DB functions
        start_time = time.time()
        try:
            if hasattr(self.server, 'pdf_server') and self.server.pdf_server:
                result = self.server.pdf_server.store_in_vector_db(pdf_path)
                duration = time.time() - start_time
                self.log_test("PDF", "vector_db", "PASS", duration, f"Stored with ID: {result}")
                print(f"  ✅ Vector DB: {duration:.2f}s")
            else:
                self.log_test("PDF", "vector_db", "SKIP", 0, "PDF server not enabled")
                print("  ⏭️  Vector DB: SKIPPED (server not enabled)")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("PDF", "vector_db", "FAIL", duration, str(e))
            print(f"  ❌ Vector DB: {duration:.2f}s - {e}")
        
        # Test 6: Knowledge Graph functions
        start_time = time.time()
        try:
            if hasattr(self.server, 'pdf_server') and self.server.pdf_server:
                result = self.server.pdf_server.create_knowledge_graph(pdf_path)
                duration = time.time() - start_time
                self.log_test("PDF", "knowledge_graph", "PASS", duration, f"Created graph with {len(result)} nodes")
                print(f"  ✅ Knowledge Graph: {duration:.2f}s")
            else:
                self.log_test("PDF", "knowledge_graph", "SKIP", 0, "PDF server not enabled")
                print("  ⏭️  Knowledge Graph: SKIPPED (server not enabled)")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("PDF", "knowledge_graph", "FAIL", duration, str(e))
            print(f"  ❌ Knowledge Graph: {duration:.2f}s - {e}")
    
    def test_audio_functions(self):
        """Test all 6 functions for Audio category"""
        print("\n🔊 Testing Audio Processing Functions...")
        
        audio_path = self.create_test_audio()
        
        # Test 1: Extract text (transcription)
        start_time = time.time()
        try:
            if hasattr(self.server, 'audio_server') and self.server.audio_server:
                result = self.server.audio_server.extract_text(audio_path)
                duration = time.time() - start_time
                self.log_test("Audio", "extract_text", "PASS", duration, f"Transcribed {len(result)} characters")
                print(f"  ✅ Extract text (transcribe): {duration:.2f}s")
            else:
                self.log_test("Audio", "extract_text", "SKIP", 0, "Audio server not enabled")
                print("  ⏭️  Extract text (transcribe): SKIPPED (server not enabled)")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Audio", "extract_text", "FAIL", duration, str(e))
            print(f"  ❌ Extract text (transcribe): {duration:.2f}s - {e}")
        
        # Test 2: Convert to image (spectrogram)
        start_time = time.time()
        try:
            if hasattr(self.server, 'audio_server') and self.server.audio_server:
                result = self.server.audio_server.convert_to_image(audio_path)
                duration = time.time() - start_time
                self.log_test("Audio", "convert_to_image", "PASS", duration, f"Generated {len(result)} spectrograms")
                print(f"  ✅ Convert to image (spectrogram): {duration:.2f}s")
            else:
                self.log_test("Audio", "convert_to_image", "SKIP", 0, "Audio server not enabled")
                print("  ⏭️  Convert to image (spectrogram): SKIPPED (server not enabled)")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Audio", "convert_to_image", "FAIL", duration, str(e))
            print(f"  ❌ Convert to image (spectrogram): {duration:.2f}s - {e}")
        
        # Test 3: Summarization
        start_time = time.time()
        try:
            if hasattr(self.server, 'audio_server') and self.server.audio_server:
                result = self.server.audio_server.summarize(audio_path)
                duration = time.time() - start_time
                self.log_test("Audio", "summarize", "PASS", duration, f"Summary length: {len(result)}")
                print(f"  ✅ Summarize: {duration:.2f}s")
            else:
                self.log_test("Audio", "summarize", "SKIP", 0, "Audio server not enabled")
                print("  ⏭️  Summarize: SKIPPED (server not enabled)")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Audio", "summarize", "FAIL", duration, str(e))
            print(f"  ❌ Summarize: {duration:.2f}s - {e}")
        
        # Test 4: Translation
        start_time = time.time()
        try:
            if hasattr(self.server, 'audio_server') and self.server.audio_server:
                result = self.server.audio_server.translate(audio_path, "en")
                duration = time.time() - start_time
                self.log_test("Audio", "translate", "PASS", duration, f"Translated to {len(result)} characters")
                print(f"  ✅ Translate: {duration:.2f}s")
            else:
                self.log_test("Audio", "translate", "SKIP", 0, "Audio server not enabled")
                print("  ⏭️  Translate: SKIPPED (server not enabled)")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Audio", "translate", "FAIL", duration, str(e))
            print(f"  ❌ Translate: {duration:.2f}s - {e}")
        
        # Test 5: Vector DB functions
        start_time = time.time()
        try:
            if hasattr(self.server, 'audio_server') and self.server.audio_server:
                result = self.server.audio_server.store_in_vector_db(audio_path)
                duration = time.time() - start_time
                self.log_test("Audio", "vector_db", "PASS", duration, f"Stored with ID: {result}")
                print(f"  ✅ Vector DB: {duration:.2f}s")
            else:
                self.log_test("Audio", "vector_db", "SKIP", 0, "Audio server not enabled")
                print("  ⏭️  Vector DB: SKIPPED (server not enabled)")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Audio", "vector_db", "FAIL", duration, str(e))
            print(f"  ❌ Vector DB: {duration:.2f}s - {e}")
        
        # Test 6: Knowledge Graph functions
        start_time = time.time()
        try:
            if hasattr(self.server, 'audio_server') and self.server.audio_server:
                result = self.server.audio_server.create_knowledge_graph(audio_path)
                duration = time.time() - start_time
                self.log_test("Audio", "knowledge_graph", "PASS", duration, f"Created graph with {len(result)} nodes")
                print(f"  ✅ Knowledge Graph: {duration:.2f}s")
            else:
                self.log_test("Audio", "knowledge_graph", "SKIP", 0, "Audio server not enabled")
                print("  ⏭️  Knowledge Graph: SKIPPED (server not enabled)")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Audio", "knowledge_graph", "FAIL", duration, str(e))
            print(f"  ❌ Knowledge Graph: {duration:.2f}s - {e}")
    
    def test_video_functions(self):
        """Test all 6 functions for Video category"""
        print("\n🎥 Testing Video Processing Functions...")
        
        video_path = self.create_test_video()
        
        # Test 1: Extract text (OCR)
        start_time = time.time()
        try:
            if hasattr(self.server, 'video_server') and self.server.video_server:
                result = self.server.video_server.extract_text(video_path)
                duration = time.time() - start_time
                self.log_test("Video", "extract_text", "PASS", duration, f"Extracted {len(result)} characters")
                print(f"  ✅ Extract text (OCR): {duration:.2f}s")
            else:
                self.log_test("Video", "extract_text", "SKIP", 0, "Video server not enabled")
                print("  ⏭️  Extract text (OCR): SKIPPED (server not enabled)")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Video", "extract_text", "FAIL", duration, str(e))
            print(f"  ❌ Extract text (OCR): {duration:.2f}s - {e}")
        
        # Test 2: Convert to image (frames)
        start_time = time.time()
        try:
            if hasattr(self.server, 'video_server') and self.server.video_server:
                result = self.server.video_server.convert_to_image(video_path)
                duration = time.time() - start_time
                self.log_test("Video", "convert_to_image", "PASS", duration, f"Extracted {len(result)} frames")
                print(f"  ✅ Convert to image (frames): {duration:.2f}s")
            else:
                self.log_test("Video", "convert_to_image", "SKIP", 0, "Video server not enabled")
                print("  ⏭️  Convert to image (frames): SKIPPED (server not enabled)")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Video", "convert_to_image", "FAIL", duration, str(e))
            print(f"  ❌ Convert to image (frames): {duration:.2f}s - {e}")
        
        # Test 3: Summarization
        start_time = time.time()
        try:
            if hasattr(self.server, 'video_server') and self.server.video_server:
                result = self.server.video_server.summarize(video_path)
                duration = time.time() - start_time
                self.log_test("Video", "summarize", "PASS", duration, f"Summary length: {len(result)}")
                print(f"  ✅ Summarize: {duration:.2f}s")
            else:
                self.log_test("Video", "summarize", "SKIP", 0, "Video server not enabled")
                print("  ⏭️  Summarize: SKIPPED (server not enabled)")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Video", "summarize", "FAIL", duration, str(e))
            print(f"  ❌ Summarize: {duration:.2f}s - {e}")
        
        # Test 4: Translation
        start_time = time.time()
        try:
            if hasattr(self.server, 'video_server') and self.server.video_server:
                result = self.server.video_server.translate(video_path, "en")
                duration = time.time() - start_time
                self.log_test("Video", "translate", "PASS", duration, f"Translated to {len(result)} characters")
                print(f"  ✅ Translate: {duration:.2f}s")
            else:
                self.log_test("Video", "translate", "SKIP", 0, "Video server not enabled")
                print("  ⏭️  Translate: SKIPPED (server not enabled)")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Video", "translate", "FAIL", duration, str(e))
            print(f"  ❌ Translate: {duration:.2f}s - {e}")
        
        # Test 5: Vector DB functions
        start_time = time.time()
        try:
            if hasattr(self.server, 'video_server') and self.server.video_server:
                result = self.server.video_server.store_in_vector_db(video_path)
                duration = time.time() - start_time
                self.log_test("Video", "vector_db", "PASS", duration, f"Stored with ID: {result}")
                print(f"  ✅ Vector DB: {duration:.2f}s")
            else:
                self.log_test("Video", "vector_db", "SKIP", 0, "Video server not enabled")
                print("  ⏭️  Vector DB: SKIPPED (server not enabled)")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Video", "vector_db", "FAIL", duration, str(e))
            print(f"  ❌ Vector DB: {duration:.2f}s - {e}")
        
        # Test 6: Knowledge Graph functions
        start_time = time.time()
        try:
            if hasattr(self.server, 'video_server') and self.server.video_server:
                result = self.server.video_server.create_knowledge_graph(video_path)
                duration = time.time() - start_time
                self.log_test("Video", "knowledge_graph", "PASS", duration, f"Created graph with {len(result)} nodes")
                print(f"  ✅ Knowledge Graph: {duration:.2f}s")
            else:
                self.log_test("Video", "knowledge_graph", "SKIP", 0, "Video server not enabled")
                print("  ⏭️  Knowledge Graph: SKIPPED (server not enabled)")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Video", "knowledge_graph", "FAIL", duration, str(e))
            print(f"  ❌ Knowledge Graph: {duration:.2f}s - {e}")
    
    def test_website_functions(self):
        """Test all 6 functions for Website category"""
        print("\n🌐 Testing Website Processing Functions...")
        
        website_path = self.create_test_website_content()
        
        # Test 1: Extract text
        start_time = time.time()
        try:
            if hasattr(self.server, 'website_server') and self.server.website_server:
                result = self.server.website_server.extract_text(website_path)
                duration = time.time() - start_time
                self.log_test("Website", "extract_text", "PASS", duration, f"Extracted {len(result)} characters")
                print(f"  ✅ Extract text: {duration:.2f}s")
            else:
                self.log_test("Website", "extract_text", "SKIP", 0, "Website server not enabled")
                print("  ⏭️  Extract text: SKIPPED (server not enabled)")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Website", "extract_text", "FAIL", duration, str(e))
            print(f"  ❌ Extract text: {duration:.2f}s - {e}")
        
        # Test 2: Convert to image (screenshot)
        start_time = time.time()
        try:
            if hasattr(self.server, 'website_server') and self.server.website_server:
                result = self.server.website_server.convert_to_image(website_path)
                duration = time.time() - start_time
                self.log_test("Website", "convert_to_image", "PASS", duration, f"Generated {len(result)} screenshots")
                print(f"  ✅ Convert to image (screenshot): {duration:.2f}s")
            else:
                self.log_test("Website", "convert_to_image", "SKIP", 0, "Website server not enabled")
                print("  ⏭️  Convert to image (screenshot): SKIPPED (server not enabled)")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Website", "convert_to_image", "FAIL", duration, str(e))
            print(f"  ❌ Convert to image (screenshot): {duration:.2f}s - {e}")
        
        # Test 3: Summarization
        start_time = time.time()
        try:
            if hasattr(self.server, 'website_server') and self.server.website_server:
                result = self.server.website_server.summarize(website_path)
                duration = time.time() - start_time
                self.log_test("Website", "summarize", "PASS", duration, f"Summary length: {len(result)}")
                print(f"  ✅ Summarize: {duration:.2f}s")
            else:
                self.log_test("Website", "summarize", "SKIP", 0, "Website server not enabled")
                print("  ⏭️  Summarize: SKIPPED (server not enabled)")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Website", "summarize", "FAIL", duration, str(e))
            print(f"  ❌ Summarize: {duration:.2f}s - {e}")
        
        # Test 4: Translation
        start_time = time.time()
        try:
            if hasattr(self.server, 'website_server') and self.server.website_server:
                result = self.server.website_server.translate(website_path, "en")
                duration = time.time() - start_time
                self.log_test("Website", "translate", "PASS", duration, f"Translated to {len(result)} characters")
                print(f"  ✅ Translate: {duration:.2f}s")
            else:
                self.log_test("Website", "translate", "SKIP", 0, "Website server not enabled")
                print("  ⏭️  Translate: SKIPPED (server not enabled)")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Website", "translate", "FAIL", duration, str(e))
            print(f"  ❌ Translate: {duration:.2f}s - {e}")
        
        # Test 5: Vector DB functions
        start_time = time.time()
        try:
            if hasattr(self.server, 'website_server') and self.server.website_server:
                result = self.server.website_server.store_in_vector_db(website_path)
                duration = time.time() - start_time
                self.log_test("Website", "vector_db", "PASS", duration, f"Stored with ID: {result}")
                print(f"  ✅ Vector DB: {duration:.2f}s")
            else:
                self.log_test("Website", "vector_db", "SKIP", 0, "Website server not enabled")
                print("  ⏭️  Vector DB: SKIPPED (server not enabled)")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Website", "vector_db", "FAIL", duration, str(e))
            print(f"  ❌ Vector DB: {duration:.2f}s - {e}")
        
        # Test 6: Knowledge Graph functions
        start_time = time.time()
        try:
            if hasattr(self.server, 'website_server') and self.server.website_server:
                result = self.server.website_server.create_knowledge_graph(website_path)
                duration = time.time() - start_time
                self.log_test("Website", "knowledge_graph", "PASS", duration, f"Created graph with {len(result)} nodes")
                print(f"  ✅ Knowledge Graph: {duration:.2f}s")
            else:
                self.log_test("Website", "knowledge_graph", "SKIP", 0, "Website server not enabled")
                print("  ⏭️  Knowledge Graph: SKIPPED (server not enabled)")
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Website", "knowledge_graph", "FAIL", duration, str(e))
            print(f"  ❌ Knowledge Graph: {duration:.2f}s - {e}")
    
    def run_all_tests(self):
        """Run all performance tests"""
        print("🚀 Starting Comprehensive Performance Testing")
        print("=" * 60)
        
        start_time = time.time()
        
        # Test all categories
        self.test_pdf_functions()
        self.test_audio_functions()
        self.test_video_functions()
        self.test_website_functions()
        
        total_time = time.time() - start_time
        
        # Generate summary report
        self.generate_report(total_time)
    
    def generate_report(self, total_time: float):
        """Generate performance test report"""
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE TEST SUMMARY")
        print("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        skipped_tests = 0
        
        for category, functions in self.results.items():
            print(f"\n📁 {category.upper()} CATEGORY:")
            for function, tests in functions.items():
                for test in tests:
                    total_tests += 1
                    if test["status"] == "PASS":
                        passed_tests += 1
                        status_icon = "✅"
                    elif test["status"] == "FAIL":
                        failed_tests += 1
                        status_icon = "❌"
                    else:  # SKIP
                        skipped_tests += 1
                        status_icon = "⏭️"
                    
                    print(f"  {status_icon} {function}: {test['duration']:.2f}s - {test['details']}")
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"  Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        print(f"  Skipped: {skipped_tests} ({skipped_tests/total_tests*100:.1f}%)")
        print(f"  Total Time: {total_time:.2f}s")
        
        # Save detailed results
        results_file = Path("Test/performance_test_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": time.time(),
                "total_time": total_time,
                "statistics": {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": failed_tests,
                    "skipped_tests": skipped_tests
                },
                "results": self.results
            }, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        if failed_tests == 0:
            print("\n🎉 All tests completed successfully!")
        else:
            print(f"\n⚠️  {failed_tests} tests failed. Check the detailed results for more information.")

def main():
    """Main test execution"""
    try:
        tester = PerformanceTester()
        tester.run_all_tests()
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())





