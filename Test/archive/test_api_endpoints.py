#!/usr/bin/env python3
"""
Test API endpoints to see what's available.
"""

import requests
import json

def test_api_endpoints():
    """Test various API endpoints."""
    base_url = "http://127.0.0.1:8001"
    
    print("🔍 Testing API endpoints...")
    print("=" * 50)
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"Root endpoint (/): {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.text[:200]}...")
    except Exception as e:
        print(f"Root endpoint error: {e}")
    
    # Test docs endpoint
    try:
        response = requests.get(f"{base_url}/docs", timeout=5)
        print(f"Docs endpoint (/docs): {response.status_code}")
    except Exception as e:
        print(f"Docs endpoint error: {e}")
    
    # Test openapi endpoint
    try:
        response = requests.get(f"{base_url}/openapi.json", timeout=5)
        print(f"OpenAPI endpoint (/openapi.json): {response.status_code}")
        if response.status_code == 200:
            openapi_data = response.json()
            paths = openapi_data.get('paths', {})
            print(f"Available paths: {list(paths.keys())}")
    except Exception as e:
        print(f"OpenAPI endpoint error: {e}")
    
    # Test specific endpoints
    endpoints_to_test = [
        "/process/pdf-enhanced-multilingual",
        "/analyze/pdf",
        "/process/pdf-to-databases",
        "/analyze/text",
        "/analyze/image",
        "/analyze/audio",
        "/analyze/video",
        "/analyze/youtube",
        "/analyze/webpage"
    ]
    
    print("\n📋 Testing specific endpoints:")
    for endpoint in endpoints_to_test:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            print(f"  {endpoint}: {response.status_code}")
        except Exception as e:
            print(f"  {endpoint}: Error - {e}")

if __name__ == "__main__":
    test_api_endpoints()
