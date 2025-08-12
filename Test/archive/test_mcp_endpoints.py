#!/usr/bin/env python3
"""
Test MCP endpoints to find the correct one for PDF processing.
"""

import requests
import json

def test_mcp_endpoints():
    """Test various MCP endpoints."""
    base_url = "http://127.0.0.1:8000/mcp"
    
    print("🔍 Testing MCP endpoints...")
    print("=" * 50)
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"Root endpoint (/): {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.text[:200]}...")
    except Exception as e:
        print(f"Root endpoint error: {e}")
    
    # Test different PDF processing endpoints
    pdf_endpoints = [
        "/process_pdf_enhanced_multilingual",
        "/process-pdf-enhanced-multilingual",
        "/pdf/process",
        "/pdf/enhanced",
        "/tools/process_pdf_enhanced_multilingual",
        "/tools/process-pdf-enhanced-multilingual"
    ]
    
    print("\n📋 Testing PDF processing endpoints:")
    for endpoint in pdf_endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            print(f"  {endpoint}: {response.status_code}")
            if response.status_code in [200, 405]:
                print(f"    └─ Available!")
        except Exception as e:
            print(f"  {endpoint}: Error - {e}")
    
    # Test POST to the process_pdf_enhanced_multilingual endpoint
    print("\n🔬 Testing POST to process_pdf_enhanced_multilingual...")
    try:
        url = f"{base_url}/process_pdf_enhanced_multilingual"
        payload = {
            "pdf_path": "data/Classical Chinese Sample 22208_0_8.pdf",
            "language": "zh",
            "generate_report": True,
            "output_path": None
        }
        
        response = requests.post(url, json=payload, timeout=10)
        print(f"POST {url}: {response.status_code}")
        if response.status_code == 200:
            print("✅ Success!")
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)[:500]}...")
        else:
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"POST error: {e}")

if __name__ == "__main__":
    test_mcp_endpoints()
