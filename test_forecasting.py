#!/usr/bin/env python3
"""
Simple test script to get detailed error messages from the forecasting endpoint
"""

import requests
import json

def test_forecasting():
    url = "http://127.0.0.1:8003/advanced-analytics/forecasting-test"
    data = {
        "data": [{"date": "2023-01-01", "sales": 100, "temperature": 20}],
        "target_variables": ["sales"],
        "forecast_horizon": 7,
        "model_type": "ensemble",
        "confidence_level": 0.95
    }
    
    try:
        response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code != 200:
            print(f"Error: {response.json()}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_forecasting()
