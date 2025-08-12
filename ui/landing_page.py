"""
Landing page for Sentiment Analysis Swarm - All Phases Overview
"""

import streamlit as st
import requests
from datetime import datetime
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Swarm - All Phases",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e, #2ca02c, #d62728, #9467bd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .phase-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .phase-card h3 {
        color: white;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .feature-list {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .quick-access {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# API configuration
API_BASE_URL = "http://localhost:8003"

def check_api_health() -> bool:
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def get_system_status() -> Dict[str, Any]:
    """Get system status from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception:
        return {}

def main():
    st.markdown('<h1 class="main-header">🧠 Sentiment Analysis Swarm</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">Complete Implementation - Phases 1-5</h2>', unsafe_allow_html=True)
    
    # System Status Check
    api_healthy = check_api_health()
    system_info = get_system_status()
    
    # Status indicator
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if api_healthy:
            st.success("✅ API Server Running (Port 8003)")
        else:
            st.error("❌ API Server Not Available")
            st.info("Please start the API server: `uvicorn src.api.main:app --host 0.0.0.0 --port 8003`")
    
    # Quick Access Section
    st.markdown('<div class="quick-access">', unsafe_allow_html=True)
    st.markdown("### 🚀 Quick Access")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📝 Text Analysis", use_container_width=True):
            st.markdown("Navigate to: http://localhost:8501")
    
    with col2:
        if st.button("📊 Dashboard", use_container_width=True):
            st.markdown("Navigate to: http://localhost:8501")
    
    with col3:
        if st.button("🤖 System Status", use_container_width=True):
            st.markdown("Navigate to: http://localhost:8501")
    
    with col4:
        if st.button("📱 Social Media", use_container_width=True):
            st.markdown("Navigate to: http://localhost:8501")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Phase Overview
    st.markdown("## 📋 Implementation Phases Overview")
    
    # Phase 1: Core Sentiment Analysis
    with st.container():
        st.markdown('<div class="phase-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Phase 1: Core Sentiment Analysis")
        st.markdown("**Status:** ✅ COMPLETED")
        st.markdown("**Duration:** 3 weeks")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Core Features:**")
            st.markdown("""
            <div class="feature-list">
            • Text sentiment analysis<br>
            • Multi-language support<br>
            • Social media analysis<br>
            • Webpage content analysis<br>
            • PDF document processing<br>
            • Audio/video analysis<br>
            • YouTube content analysis
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Technical Achievements:**")
            st.markdown("""
            <div class="feature-list">
            • 10 specialized agents<br>
            • MCP server integration<br>
            • FastAPI REST endpoints<br>
            • Agent swarm architecture<br>
            • Real-time processing<br>
            • Multi-modal analysis
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Phase 2: Business Intelligence
    with st.container():
        st.markdown('<div class="phase-card">', unsafe_allow_html=True)
        st.markdown("### 💼 Phase 2: Business Intelligence")
        st.markdown("**Status:** ✅ COMPLETED")
        st.markdown("**Duration:** 2 weeks")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Business Features:**")
            st.markdown("""
            <div class="feature-list">
            • Executive dashboards<br>
            • Business summaries<br>
            • Data visualizations<br>
            • Market trend analysis<br>
            • Financial data integration<br>
            • News monitoring<br>
            • Cross-modal insights
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Integration Capabilities:**")
            st.markdown("""
            <div class="feature-list">
            • Social media APIs<br>
            • Database connections<br>
            • External API fetching<br>
            • Real-time data streams<br>
            • Automated reporting<br>
            • Actionable insights
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Phase 3: Advanced Analytics
    with st.container():
        st.markdown('<div class="phase-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Phase 3: Advanced Analytics")
        st.markdown("**Status:** ✅ COMPLETED")
        st.markdown("**Duration:** 2 weeks")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Advanced Features:**")
            st.markdown("""
            <div class="feature-list">
            • Comprehensive analysis<br>
            • Cross-modal insights<br>
            • Business intelligence reports<br>
            • Content storytelling<br>
            • Data storytelling<br>
            • Actionable insights<br>
            • Performance optimization
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Analytics Capabilities:**")
            st.markdown("""
            <div class="feature-list">
            • Multi-dimensional analysis<br>
            • Pattern recognition<br>
            • Trend prediction<br>
            • Anomaly detection<br>
            • Statistical modeling<br>
            • Real-time analytics
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Phase 4: Export & Automation
    with st.container():
        st.markdown('<div class="phase-card">', unsafe_allow_html=True)
        st.markdown("### 🔄 Phase 4: Export & Automation")
        st.markdown("**Status:** ✅ COMPLETED")
        st.markdown("**Duration:** 2 weeks")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Export Features:**")
            st.markdown("""
            <div class="feature-list">
            • Multi-format export (PDF, Excel, CSV)<br>
            • Automated report generation<br>
            • Scheduled reporting<br>
            • Report sharing capabilities<br>
            • Data export history<br>
            • Custom report templates
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Automation Capabilities:**")
            st.markdown("""
            <div class="feature-list">
            • Automated workflows<br>
            • Scheduled tasks<br>
            • Batch processing<br>
            • Email notifications<br>
            • Report distribution<br>
            • Integration APIs
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Phase 5: Semantic Search & Reflection
    with st.container():
        st.markdown('<div class="phase-card">', unsafe_allow_html=True)
        st.markdown("### 🧠 Phase 5: Semantic Search & Reflection")
        st.markdown("**Status:** ✅ COMPLETED")
        st.markdown("**Duration:** 2 weeks")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Semantic Search:**")
            st.markdown("""
            <div class="feature-list">
            • Cross-modal semantic search<br>
            • Intelligent query routing<br>
            • Result combination<br>
            • Agent capability discovery<br>
            • Multi-agent coordination<br>
            • Accuracy optimization
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Agent Reflection:**")
            st.markdown("""
            <div class="feature-list">
            • Centralized reflection coordinator<br>
            • Real-time agent questioning<br>
            • Response validation<br>
            • Quality assessment<br>
            • Continuous improvement<br>
            • Self-optimization
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # System Statistics
    if system_info:
        st.markdown("## 📊 System Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("API Endpoints", len(system_info.get("endpoints", {})))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("System Version", system_info.get("version", "1.0.0"))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Phases", "5")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Implementation Time", "11 weeks")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Available Endpoints
    if system_info and "endpoints" in system_info:
        st.markdown("## 🔗 Available API Endpoints")
        
        # Group endpoints by category
        categories = {
            "Core Analysis": ["text_analysis", "image_analysis", "video_analysis", "audio_analysis", "webpage_analysis", "pdf_analysis"],
            "Business Intelligence": ["business_dashboard", "executive_summary", "data_visualizations", "business_trends"],
            "Export & Automation": ["export_analysis_results", "generate_automated_reports", "share_reports", "schedule_reports"],
            "Semantic Search": ["semantic_search", "query_routing", "result_combination", "agent_capabilities"],
            "Agent Reflection": ["agent_reflection", "agent_questioning", "reflection_insights", "response_validation"]
        }
        
        for category, endpoint_keys in categories.items():
            with st.expander(f"📁 {category} ({len(endpoint_keys)} endpoints)"):
                for key in endpoint_keys:
                    if key in system_info["endpoints"]:
                        st.code(f"GET/POST {system_info['endpoints'][key]}")
    
    # Footer
    st.markdown("---")
    st.markdown(f"<p style='text-align: center; color: #666;'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>🧠 Sentiment Analysis Swarm - Complete Implementation</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
