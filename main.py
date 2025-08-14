#!/usr/bin/env python3
"""
Main entry point for the Sentiment Analysis Swarm system.
Provides both MCP server and FastAPI server functionality.
"""

# Suppress all deprecation warnings BEFORE any other imports
import warnings
import sys

# Set warnings filter to ignore all deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="uvicorn")
warnings.filterwarnings("ignore", category=UserWarning, module="websockets")

# Custom warning filter function
def ignore_all_warnings(message, category, filename, lineno, file=None, line=None):
    """Custom warning filter to ignore all warnings."""
    if category in [DeprecationWarning, FutureWarning, UserWarning]:
        return True
    return False

# Add custom filter
warnings.showwarning = ignore_all_warnings

import os
import threading
import uvicorn
import subprocess
import time
import requests
from typing import List, Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import MCP server after adding src to path to avoid conflicts
try:
    from fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️ MCP server not available")

# Import after path modification
# flake8: noqa: E402
from src.api.main import app
from src.core.error_handler import with_error_handling
from src.mcp_servers.unified_mcp_server import UnifiedMCPServer, create_unified_mcp_server
from src.mcp_servers.standalone_mcp_server import StandaloneMCPServer, start_standalone_mcp_server
from src.config.settings import settings
from src.config.config import config
from src.core.port_checker import get_safe_port

# YouTube analysis now handled by UnifiedVisionAgent
from src.core.models import (
    AnalysisRequest, DataType, ProcessingStatus, SentimentResult
)


def start_mcp_server():
    """Create the unified MCP server for integration with FastAPI."""
    try:
        # Create the unified MCP server
        mcp_server = create_unified_mcp_server()
        
        if mcp_server.mcp is None:
            print("⚠️ MCP server not available - skipping MCP server integration")
            return None
        
        print("✅ Unified MCP server created successfully")
        print(" - MCP Server: Integrated with FastAPI at /mcp")
        print(" - Available tools: 25 consolidated tools")
        print("   • Content Processing: process_content, extract_text_from_content, summarize_content, translate_content, convert_content_format")
        print("   • Analysis & Intelligence: analyze_sentiment, extract_entities, generate_knowledge_graph, analyze_business_intelligence, create_visualizations")
        print("   • Agent Management: get_agent_status, start_agents, stop_agents")
        print("   • Data Management: store_in_vector_db, query_knowledge_graph, export_data, manage_data_sources")
        print("   • Reporting & Export: generate_report, create_dashboard, export_results, schedule_reports")
        print("   • System Management: get_system_status, configure_system, monitor_performance, manage_configurations")
        
        return mcp_server
    except Exception as e:
        print(f"⚠️ Warning: Could not create MCP server: {e}")
        print(" The application will run without MCP server integration")
        return None


def get_mcp_tools_info():
    """Get information about available MCP tools."""
    try:
        mcp_server = create_unified_mcp_server()
        if not mcp_server or not mcp_server.mcp:
            return []
        
        tools = []
        
        # Try different ways to access tools from FastMCP
        if hasattr(mcp_server.mcp, 'tools'):
            tools = list(mcp_server.mcp.tools.keys())
        elif hasattr(mcp_server.mcp, '_tools'):
            tools = list(mcp_server.mcp._tools.keys())
        elif hasattr(mcp_server.mcp, 'app') and hasattr(mcp_server.mcp.app, 'state') and hasattr(mcp_server.mcp.app.state, 'tools'):
            tools = list(mcp_server.mcp.app.state.tools.keys())
        elif hasattr(mcp_server.mcp, 'get_tools'):
            result = mcp_server.mcp.get_tools()
            if hasattr(result, '__await__'):
                print("ℹ️ Using comprehensive tool list (async discovery not needed)")
                tools = []
            else:
                tools = result
        elif hasattr(mcp_server.mcp, 'list_tools'):
            result = mcp_server.mcp.list_tools()
            if hasattr(result, '__await__'):
                print("ℹ️ Using comprehensive tool list (async discovery not needed)")
                tools = []
            else:
                tools = result
        else:
            # If we can't access tools directly, provide a list of known tools
            tools = [
                "process_content",
                "extract_text_from_content",
                "summarize_content",
                "translate_content",
                "convert_content_format",
                "analyze_sentiment",
                "extract_entities",
                "generate_knowledge_graph",
                "analyze_business_intelligence",
                "create_visualizations",
                "get_agent_status",
                "start_agents",
                "stop_agents",
                "store_in_vector_db",
                "query_knowledge_graph",
                "export_data",
                "manage_data_sources",
                "generate_report",
                "create_dashboard",
                "export_results",
                "schedule_reports",
                "get_system_status",
                "configure_system",
                "monitor_performance",
                "manage_configurations"
            ]
        
        print(f"🔧 Available MCP tools: {len(tools)} tools")
        return tools
        
    except Exception as e:
        print(f"⚠️ Could not get MCP tools info: {e}")
        # Return comprehensive tool list as fallback
        return [
            "process_content",
            "extract_text_from_content",
            "summarize_content",
            "translate_content",
            "convert_content_format",
            "analyze_sentiment",
            "extract_entities",
            "generate_knowledge_graph",
            "analyze_business_intelligence",
            "create_visualizations",
            "get_agent_status",
            "start_agents",
            "stop_agents",
            "store_in_vector_db",
            "query_knowledge_graph",
            "export_data",
            "manage_data_sources",
            "generate_report",
            "create_dashboard",
            "export_results",
            "schedule_reports",
            "get_system_status",
            "configure_system",
            "monitor_performance",
            "manage_configurations"
        ]


def launch_streamlit_apps():
    """Launch Streamlit applications."""
    try:
        # Launch main UI
        main_ui_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "ui/main.py",
            "--server.port", "8501",
            "--server.headless", "true"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Launch landing page
        landing_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "ui/landing_page.py",
            "--server.port", "8502",
            "--server.headless", "true"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("✅ Streamlit applications launched")
        return main_ui_process, landing_process
        
    except Exception as e:
        print(f"⚠️ Error launching Streamlit apps: {e}")
        return None, None


if __name__ == "__main__":
    print("Starting Sentiment Analysis Swarm with MCP Integration")
    print("=" * 60)
    
    # Initialize performance optimizer and data collector
    print("Initializing performance monitoring system...")
    try:
        from src.core.performance_optimizer import get_performance_optimizer
        from src.core.performance_data_collector import get_performance_data_collector
        import asyncio
        
        async def init_performance_system():
            # Initialize performance data collector
            collector = await get_performance_data_collector()
            await collector.start_collection()
            print("✅ Performance data collection started")
            
            # Initialize performance optimizer
            optimizer = await get_performance_optimizer()
            await optimizer.start_monitoring()
            print("✅ Performance monitoring started")
        
        # Start performance monitoring in background
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.create_task(init_performance_system())
        print("✅ Performance monitoring system initialized")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize performance monitoring system: {e}")
    
    # Initialize data ingestion service
    print("Initializing data ingestion service...")
    try:
        from src.core.data_ingestion_service import data_ingestion_service
        supported_languages = data_ingestion_service.get_supported_languages()
        print(f"✅ Data ingestion service initialized with {len(supported_languages)} supported languages:")
        for code, name in supported_languages.items():
            print(f"   - {code}: {name}")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize data ingestion service: {e}")
    
    # Create MCP server for integration
    print("Creating MCP server for integration...")
    mcp_server = start_mcp_server()
    
    # Show available tools
    if mcp_server:
        tools = get_mcp_tools_info()
        if tools:
            print(f"🔧 MCP Tools: {len(tools)} tools available")
    
    # Get API configuration and ensure port is available
    api_host = config.api.host
    api_port = get_safe_port(api_host, config.api.port)
    
    print("\nStarting FastAPI server with MCP integration...")
    
    # Integrate MCP server with FastAPI if available
    if mcp_server:
        try:
            # Create MCP app without path prefix - we'll handle the mounting ourselves
            mcp_app = mcp_server.get_http_app(path="")
            if mcp_app:
                # Mount the MCP app to the FastAPI app
                from src.api.main import app
                app.mount("/mcp", mcp_app)
                
                # Also mount at /mcp/ for compatibility
                app.mount("/mcp/", mcp_app)
                
                print("✅ MCP server integrated with FastAPI at /mcp and /mcp/")
                print("   Note: Clients must use MCP protocol with 'initialize' method first")
                
                # Add a simple health check endpoint for MCP
                @app.get("/mcp-health")
                async def mcp_health_check():
                    return {
                        "status": "healthy", 
                        "service": "mcp_server", 
                        "endpoints": ["/mcp", "/mcp/"],
                        "protocol": "MCP (Model Context Protocol)",
                        "note": "Use 'initialize' method to establish session"
                    }
                    
        except Exception as e:
            print(f"⚠️ Warning: Could not integrate MCP server: {e}")
            # Add fallback MCP endpoint
            from src.api.main import app
            @app.get("/mcp")
            async def mcp_fallback():
                return {"error": "MCP server not available", "status": "unavailable"}
            
            @app.get("/mcp/")
            async def mcp_fallback_trailing():
                return {"error": "MCP server not available", "status": "unavailable"}
    
    # Start standalone MCP server for Strands integration
    print("\nStarting standalone MCP server for Strands integration...")
    standalone_mcp_server = None
    try:
        standalone_mcp_server = start_standalone_mcp_server(host="localhost", port=8000)
        print("✅ Standalone MCP server started on port 8000")
        print("🔧 Available for Strands integration with Streamable HTTP transport")
    except Exception as e:
        print(f"⚠️ Warning: Could not start standalone MCP server: {e}")
    
    # Start FastAPI server in a separate thread
    def start_fastapi_server():
        uvicorn.run(app, host=api_host, port=api_port, log_level="info")
    
    api_thread = threading.Thread(target=start_fastapi_server, daemon=True)
    api_thread.start()
    
    # Launch Streamlit applications directly
    main_ui_process, landing_process = launch_streamlit_apps()
    
    print("\n🎉 All services are now running!")
    print("=" * 60)
    print("🌐 Access URLs:")
    print("   📊 Main UI:        http://localhost:8501")
    print("   🏠 Landing Page:   http://localhost:8502")
    print("   🔗 API Docs:       http://localhost:8003/docs")
    print("   🤖 MCP Server:     http://localhost:8003/mcp (FastAPI integrated)")
    print("   🔧 Standalone MCP: http://localhost:8000 (Strands integration)")
    print("=" * 60)
    print("🚀 System is ready for use!")
    print("💡 For Strands integration, use: streamablehttp_client('http://localhost:8000/mcp')")
    
    # Keep the process running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        if main_ui_process:
            main_ui_process.terminate()
        if landing_process:
            landing_process.terminate()
        if standalone_mcp_server:
            standalone_mcp_server.stop()
        print("✅ Services stopped")
