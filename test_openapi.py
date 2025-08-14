#!/usr/bin/env python3
"""
Test OpenAPI Generation
"""

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# Simple test app
app = FastAPI(
    title="Test API",
    description="Test API for OpenAPI generation",
    version="1.0.0"
)

class TestRequest(BaseModel):
    content: str
    language: str = "en"

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/test")
async def test_endpoint(request: TestRequest):
    return {"result": f"Processed: {request.content}"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8004)

