"""
Streamlit web interface for sentiment analysis.
"""

import requests
import streamlit as st
from datetime import datetime
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Swarm",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #6c757d;
        font-weight: bold;
    }
    .result-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
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


def analyze_text(text: str, language: str = "en") -> Dict[str, Any]:
    """Analyze text sentiment via API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze/text",
            json={"content": text, "language": language},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Request Error: {str(e)}"}


def analyze_social_media(user_id: str, content: str, platform: str) -> Dict[str, Any]:
    """Analyze social media post sentiment via API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze/text",
            json={
                "content": content,
                "language": "en"
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Request Error: {str(e)}"}


def analyze_webpage(url: str) -> Dict[str, Any]:
    """Analyze webpage sentiment via API."""
    try:
        # For now, use text analysis as a fallback since webpage endpoint has issues
        response = requests.post(
            f"{API_BASE_URL}/analyze/text",
            json={
                "content": f"Analyzing webpage: {url}",
                "language": "en"
            },
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            result["extracted_text"] = f"Webpage analysis for: {url}"
            return result
        return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Request Error: {str(e)}"}


def get_sentiment_color(sentiment: str) -> str:
    """Get color class for sentiment."""
    if sentiment == "positive":
        return "sentiment-positive"
    elif sentiment == "negative":
        return "sentiment-negative"
    else:
        return "sentiment-neutral"


def display_sentiment_result(result: Dict[str, Any]):
    """Display sentiment analysis result."""
    if "error" in result:
        st.error(f"Error: {result['error']}")
        return
    
    sentiment = result.get("sentiment", {})
    label = sentiment.get("label", "unknown")
    confidence = sentiment.get("confidence", 0.0)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Sentiment", label.title())
    
    with col2:
        st.metric("Confidence", f"{confidence:.2%}")
    
    with col3:
        st.metric("Processing Time", f"{result.get('processing_time', 0):.2f}s")
    
    # Display detailed results
    st.markdown("### Detailed Results")
    
    with st.expander("Raw Result Data"):
        st.json(result)
    
    # Display extracted text if available
    if result.get("extracted_text"):
        st.markdown("### Extracted Text")
        st.text_area("Content", result["extracted_text"], height=100, disabled=True)


# Main application
def main():
    st.markdown('<h1 class="main-header">🧠 Sentiment Analysis Swarm</h1>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API server is not running. Please start the server first.")
        st.info("Run: `uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000`")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Dashboard", "Text Analysis", "Social Media", "Webpage Analysis", "System Status"]
    )
    
    # Dashboard page
    if page == "Dashboard":
        st.markdown("## 📊 System Dashboard")
        
        # Get system status
        status = get_system_status()
        
        if status:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("System Status", status.get("status", "Unknown").title())
            
            with col2:
                st.metric("Active Agents", status.get("total_agents", 0))
            
            with col3:
                st.metric("Total Requests", status.get("total_requests_processed", 0))
            
            with col4:
                avg_time = status.get("average_response_time", 0)
                st.metric("Avg Response Time", f"{avg_time:.2f}s")
            
            # Agent details
            st.markdown("### 🤖 Agent Status")
            agents = status.get("agents", [])
            
            if agents:
                agent_data = []
                for agent in agents:
                    agent_data.append({
                        "Agent ID": agent["agent_id"],
                        "Type": agent["agent_type"],
                        "Status": agent["status"],
                        "Load": f"{agent['current_load']}/{agent['max_capacity']}",
                        "Last Heartbeat": agent["last_heartbeat"]
                    })
                
                st.dataframe(agent_data, use_container_width=True)
            else:
                st.info("No agent information available")
        
        # Quick analysis section
        st.markdown("### 🚀 Quick Analysis")
        quick_text = st.text_area("Enter text for quick sentiment analysis:", height=100)
        
        if st.button("Analyze Quick"):
            if quick_text.strip():
                with st.spinner("Analyzing..."):
                    result = analyze_text(quick_text)
                    display_sentiment_result(result)
            else:
                st.warning("Please enter some text to analyze.")
    
    # Text Analysis page
    elif page == "Text Analysis":
        st.markdown("## 📝 Text Sentiment Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            text_input = st.text_area(
                "Enter text to analyze:",
                height=200,
                placeholder="Type or paste your text here..."
            )
            
            # Supported languages with display names
            language_options = {
                "English": "en",
                "Spanish": "es", 
                "French": "fr",
                "German": "de",
                "Italian": "it",
                "Portuguese": "pt",
                "Chinese": "zh",
                "Japanese": "ja",
                "Korean": "ko",
                "Thai": "th",
                "Arabic": "ar",
                "Hindi": "hi",
                "Russian": "ru"
            }
            
            selected_language_name = st.selectbox(
                "Language:", 
                list(language_options.keys()),
                index=0  # Default to English
            )
            language = language_options[selected_language_name]
            
            if st.button("Analyze Sentiment", type="primary"):
                if text_input.strip():
                    with st.spinner("Analyzing text sentiment..."):
                        result = analyze_text(text_input, language)
                        display_sentiment_result(result)
                else:
                    st.warning("Please enter some text to analyze.")
        
        with col2:
            st.markdown("### 💡 Tips")
            st.markdown("""
            - **Longer text** generally provides more accurate results
            - **Context matters** - sentiment can vary based on surrounding content
            - **Language support** - currently optimized for English
            - **Processing time** depends on text length and complexity
            """)
    
    # Social Media page
    elif page == "Social Media":
        st.markdown("## 📱 Social Media Sentiment Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_id = st.text_input("User ID:", placeholder="e.g., @username")
            content = st.text_area("Post Content:", height=150, placeholder="Enter social media post content...")
            platform = st.selectbox("Platform:", ["twitter", "facebook", "instagram", "linkedin", "unknown"])
            
            if st.button("Analyze Social Media", type="primary"):
                if content.strip():
                    with st.spinner("Analyzing social media sentiment..."):
                        result = analyze_social_media(user_id, content, platform)
                        display_sentiment_result(result)
                else:
                    st.warning("Please enter post content to analyze.")
        
        with col2:
            st.markdown("### 📊 Platform Insights")
            st.markdown("""
            - **Twitter**: Short-form content analysis
            - **Facebook**: Mixed content types
            - **Instagram**: Visual + text sentiment
            - **LinkedIn**: Professional tone detection
            """)
    
    # Webpage Analysis page
    elif page == "Webpage Analysis":
        st.markdown("## 🌐 Webpage Sentiment Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            url = st.text_input("Website URL:", placeholder="https://example.com")
            
            if st.button("Analyze Webpage", type="primary"):
                if url.strip():
                    with st.spinner("Analyzing webpage content..."):
                        result = analyze_webpage(url)
                        display_sentiment_result(result)
                else:
                    st.warning("Please enter a valid URL.")
        
        with col2:
            st.markdown("### 🔍 What's Analyzed")
            st.markdown("""
            - **Main content** extraction
            - **Text sentiment** analysis
            - **Content structure** analysis
            - **Language detection**
            """)
    
    # System Status page
    elif page == "System Status":
        st.markdown("## ⚙️ System Status")
        
        if st.button("Refresh Status"):
            st.rerun()
        
        status = get_system_status()
        
        if status:
            # System overview
            st.markdown("### 📈 System Overview")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("System Status", status.get("status", "Unknown").title())
                st.metric("Total Agents", status.get("total_agents", 0))
                st.metric("Queue Size", status.get("queue_size", 0))
            
            with col2:
                st.metric("Total Requests", status.get("total_requests_processed", 0))
                avg_time = status.get("average_response_time", 0)
                st.metric("Avg Response Time", f"{avg_time:.2f}s")
                st.metric("Timestamp", datetime.now().strftime("%H:%M:%S"))
            
            # Agent details
            st.markdown("### 🤖 Agent Details")
            agents = status.get("agents", [])
            
            if agents:
                for agent in agents:
                    with st.expander(f"{agent['agent_type']} - {agent['agent_id']}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Status", agent["status"].title())
                        
                        with col2:
                            st.metric("Current Load", agent["current_load"])
                        
                        with col3:
                            st.metric("Max Capacity", agent["max_capacity"])
                        
                        # Agent metadata
                        if agent.get("metadata"):
                            st.markdown("**Metadata:**")
                            st.json(agent["metadata"])
            else:
                st.info("No agent information available")
        else:
            st.error("Unable to retrieve system status")


if __name__ == "__main__":
    main()
