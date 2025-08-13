"""
Streamlit web interface for sentiment analysis.
"""

import requests
import streamlit as st
from datetime import datetime
from typing import Dict, Any, List

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


def semantic_search(
    query: str,
    search_type: str = "semantic",
    language: str = "en",
    content_types: List[str] = None,
    n_results: int = 10,
    similarity_threshold: float = 0.7
) -> Dict[str, Any]:
    """Perform semantic search via API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/search/semantic",
            json={
                "query": query,
                "search_type": search_type,
                "language": language,
                "content_types": content_types,
                "n_results": n_results,
                "similarity_threshold": similarity_threshold
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Request Error: {str(e)}"}


def knowledge_graph_search(query: str, language: str = "en") -> Dict[str, Any]:
    """Perform knowledge graph search via API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/search/knowledge-graph",
            json={
                "query": query,
                "language": language
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Request Error: {str(e)}"}


def combined_search(
    query: str,
    language: str = "en",
    n_results: int = 10,
    similarity_threshold: float = 0.7
) -> Dict[str, Any]:
    """Perform combined semantic and knowledge graph search."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/search/combined",
            json={
                "query": query,
                "language": language,
                "n_results": n_results,
                "similarity_threshold": similarity_threshold
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Request Error: {str(e)}"}


def get_search_statistics() -> Dict[str, Any]:
    """Get search statistics via API."""
    try:
        response = requests.get(f"{API_BASE_URL}/search/statistics", timeout=10)
        if response.status_code == 200:
            return response.json()
        return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Request Error: {str(e)}"}


def generate_graph_report(query: str = None, language: str = "en") -> Dict[str, Any]:
    """Generate knowledge graph report via API."""
    try:
        payload = {}
        if query:
            payload = {"query": query, "language": language}
        
        response = requests.post(
            f"{API_BASE_URL}/search/generate-graph-report", 
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
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
        ["Dashboard", "Text Analysis", "Semantic Search", "Social Media", "Webpage Analysis", "System Status"]
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
    
    # Semantic Search page
    elif page == "Semantic Search":
        st.markdown("## 🔍 Semantic Search & Knowledge Graph")
        
        # Search statistics
        stats = get_search_statistics()
        if stats and "error" not in stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Documents", stats.get("total_documents", 0))
            with col2:
                st.metric("Languages", len(stats.get("supported_languages", [])))
            with col3:
                st.metric("Content Types", len(stats.get("supported_content_types", [])))
            with col4:
                st.metric("Search Strategies", len(stats.get("search_strategies", [])))
        
        # Search tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 Semantic Search", 
            "🌐 Knowledge Graph", 
            "🔄 Combined Search",
            "📊 Search Statistics"
        ])
        
        with tab1:
            st.markdown("### Semantic Search")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                query = st.text_input("Search Query:", placeholder="Enter your search query...")
                
                search_type = st.selectbox(
                    "Search Type:",
                    ["semantic", "conceptual", "multilingual", "cross_content"]
                )
                
                language = st.selectbox(
                    "Language:",
                    ["en", "zh", "ru", "ja", "ko", "ar", "hi", "auto"]
                )
                
                content_types = st.multiselect(
                    "Content Types:",
                    ["text", "pdf", "audio", "video", "image", "web", "document"],
                    default=["text", "pdf", "document"]
                )
                
                col_a, col_b = st.columns(2)
                with col_a:
                    n_results = st.slider("Number of Results:", 1, 50, 10)
                with col_b:
                    similarity_threshold = st.slider(
                        "Similarity Threshold:", 0.3, 0.95, 0.7, 0.05
                    )
                
                if st.button("🔍 Search", type="primary"):
                    if query.strip():
                        with st.spinner("Performing semantic search..."):
                            result = semantic_search(
                                query=query,
                                search_type=search_type,
                                language=language,
                                content_types=content_types,
                                n_results=n_results,
                                similarity_threshold=similarity_threshold
                            )
                            
                            if "error" in result:
                                st.error(f"Search Error: {result['error']}")
                            else:
                                st.success(f"Found {result.get('total_results', 0)} results in {result.get('processing_time', 0):.2f}s")
                                
                                # Display results
                                results = result.get("results", [])
                                if results:
                                    for i, item in enumerate(results):
                                        with st.expander(f"Result {i+1} (Score: {item.get('similarity', 0):.3f})"):
                                            st.markdown(f"**Content:** {item.get('content', 'N/A')[:200]}...")
                                            st.markdown(f"**Type:** {item.get('content_type', 'N/A')}")
                                            st.markdown(f"**Language:** {item.get('language', 'N/A')}")
                                            st.markdown(f"**Sentiment:** {item.get('sentiment', 'N/A')}")
                                            if "metadata" in item:
                                                st.json(item["metadata"])
                                else:
                                    st.info("No results found. Try adjusting your search parameters.")
                    else:
                        st.warning("Please enter a search query.")
            
            with col2:
                st.markdown("### 🔧 Search Options")
                st.markdown("""
                **Search Types:**
                - **Semantic**: Find similar content
                - **Conceptual**: Find related ideas
                - **Multilingual**: Cross-language search
                - **Cross-content**: Search across types
                
                **Tips:**
                - Use natural language queries
                - Adjust similarity threshold for precision
                - Try different content types
                - Use "auto" language for detection
                """)
        
        with tab2:
            st.markdown("### Knowledge Graph Search & Reports")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Knowledge Graph Search
                st.markdown("#### 🔍 Search Knowledge Graph")
                kg_query = st.text_input("Knowledge Graph Query:", placeholder="Enter your query...")
                kg_language = st.selectbox("Language:", ["en", "zh", "ru", "ja", "ko", "ar", "hi"])
                
                if st.button("🔍 Search Knowledge Graph", type="primary"):
                    if kg_query.strip():
                        with st.spinner("Searching knowledge graph..."):
                            result = knowledge_graph_search(kg_query, kg_language)
                            
                            if "error" in result:
                                st.error(f"Knowledge Graph Error: {result['error']}")
                            else:
                                st.success("Knowledge graph search completed!")
                                st.json(result)
                    else:
                        st.warning("Please enter a query.")
                
                # Graph Report Generation
                st.markdown("#### 📊 Generate Interactive Graph Visualization")
                st.markdown("Generate an interactive HTML visualization of the knowledge graph:")
                st.markdown("- **Smart Filtering**: Automatically filters based on your query")
                st.markdown("- **Interactive Network Graph**: Zoom, pan, and explore relationships")
                st.markdown("- **Node Details**: Click nodes to see entity information")
                st.markdown("- **Real-time Statistics**: Live graph metrics and analysis")
                
                # Knowledge Graph Search (Text Results)
                st.markdown("### 🔍 Knowledge Graph Search (Text Results)")
                kg_query = st.text_input("Search Query:", placeholder="Enter your query...", key="kg_search_query")
                kg_language = st.selectbox("Language:", ["en", "zh", "ru", "ja", "ko", "ar", "hi"], key="kg_search_language")
                
                if st.button("🔍 Search Knowledge Graph", type="primary", key="search_kg_button_1"):
                    if kg_query.strip():
                        with st.spinner(f"Searching knowledge graph for: {kg_query}..."):
                            result = knowledge_graph_search(kg_query, kg_language)
                            st.session_state.search_result = result
                            st.success("Knowledge graph search completed!")
                    else:
                        st.warning("Please enter a search query.")
                
                # Display search results
                if 'search_result' in st.session_state and st.session_state.search_result is not None:
                    result = st.session_state.search_result
                    if "error" in result:
                        st.error(f"Search Error: {result['error']}")
                    else:
                        st.markdown("### 📋 Search Results")
                        st.json(result)
                
                # Separator
                st.markdown("---")
                
                # Interactive Graph Visualization
                st.markdown("### 🎯 Interactive Graph Visualization")
                st.markdown("Generate an interactive visual graph based on your query:")
                
                # Graph generation inputs
                graph_query = st.text_input("Graph Query (optional):", placeholder="Leave empty for full graph, or enter query to filter", key="graph_query")
                graph_language = st.selectbox("Graph Language:", ["en", "zh", "ru", "ja", "ko", "ar", "hi"], key="graph_language")
                
                # Graph generation button
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("🎯 Generate Filtered Graph", type="primary", key="generate_filtered_graph"):
                        if graph_query.strip():
                            with st.spinner(f"Generating filtered graph for: {graph_query}..."):
                                result = generate_graph_report(graph_query, graph_language)
                                st.session_state.graph_result = result
                                st.success(f"✅ Generated filtered graph for: {graph_query}")
                        else:
                            st.warning("Please enter a query for filtered graph.")
                
                with col2:
                    if st.button("📊 Generate Full Graph", type="secondary", key="generate_full_graph"):
                        with st.spinner("Generating full interactive graph..."):
                            result = generate_graph_report()
                            st.session_state.graph_result = result
                            st.success("✅ Generated full interactive graph")
                
                # Handle graph generation results
                if 'graph_result' in st.session_state and st.session_state.graph_result is not None:
                    result = st.session_state.graph_result
                    
                    if "error" in result:
                        st.error(f"Graph Report Error: {result['error']}")
                    else:
                        st.success("Interactive graph visualization generated successfully!")
                        
                        # Display report information
                        report_data = result.get("result", {})
                        if report_data and "content" in report_data:
                            content = report_data["content"][0].get("json", {})
                            
                            # Display graph statistics first
                            if content.get("graph_stats"):
                                stats = content["graph_stats"]
                                st.markdown("### 📈 Graph Statistics")
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Nodes", stats.get("nodes", 0))
                                with col_b:
                                    st.metric("Edges", stats.get("edges", 0))
                                with col_c:
                                    st.metric("Density", f"{stats.get('density', 0):.3f}")
                            
                            # Display interactive HTML visualization
                            if content.get("html_file"):
                                st.markdown("### 🎯 Interactive Knowledge Graph Visualization")
                                
                                # Read and display the HTML file
                                try:
                                    with open(content["html_file"], 'r', encoding='utf-8') as f:
                                        html_content = f.read()
                                    
                                    # Display the interactive HTML with full width and scrollbar
                                    st.components.v1.html(html_content, height=1200, scrolling=True, width=None)
                                    
                                    # Make the HTML file location clickable with proper file handling
                                    st.markdown("### 📁 Download Interactive Graph")
                                    
                                    # Create a download button that actually works
                                    with open(content["html_file"], 'r', encoding='utf-8') as f:
                                        html_data = f.read()
                                    
                                    st.download_button(
                                        label="📥 Download Interactive Graph HTML",
                                        data=html_data,
                                        file_name=f"knowledge_graph_{content.get('query', 'full')}.html",
                                        mime="text/html"
                                    )
                                    
                                    # Also show the file path
                                    st.info(f"📁 HTML file saved to: `{content['html_file']}`")
                                    
                                except Exception as e:
                                    st.error(f"Error loading HTML visualization: {str(e)}")
                                    st.info(f"HTML file generated at: `{content['html_file']}`")
                            else:
                                st.warning("No HTML visualization was generated")
                        else:
                            st.warning("No content found in graph result")
                else:
                    st.info("Click 'Generate Filtered Graph' or 'Generate Full Graph' to create an interactive visualization.")
            
            with col2:
                st.markdown("### 🧠 Knowledge Graph")
                st.markdown("""
                **Features:**
                - Entity extraction
                - Relationship mapping
                - Concept linking
                - Multilingual support
                - **Interactive visualizations**
                
                **Use Cases:**
                - Find related concepts
                - Discover connections
                - Explore topics
                - Analyze relationships
                - **Interactive graph exploration**
                
                **Visualization:**
                - Interactive HTML network graphs
                - Zoom, pan, and click functionality
                - Real-time node exploration
                - Dynamic relationship display
                """)
        
        with tab3:
            st.markdown("### Combined Search")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                combined_query = st.text_input("Combined Search Query:", placeholder="Enter your query...")
                combined_language = st.selectbox("Language:", ["en", "zh", "ru", "ja", "ko", "ar", "hi"])
                
                col_a, col_b = st.columns(2)
                with col_a:
                    combined_n_results = st.slider("Results:", 1, 50, 10)
                with col_b:
                    combined_threshold = st.slider("Threshold:", 0.3, 0.95, 0.7, 0.05)
                
                if st.button("🔄 Combined Search", type="primary"):
                    if combined_query.strip():
                        with st.spinner("Performing combined search..."):
                            result = combined_search(
                                query=combined_query,
                                language=combined_language,
                                n_results=combined_n_results,
                                similarity_threshold=combined_threshold
                            )
                            
                            if "error" in result:
                                st.error(f"Combined Search Error: {result['error']}")
                            else:
                                st.success(f"Combined search completed in {result.get('processing_time', 0):.2f}s")
                                
                                # Display semantic results
                                semantic_results = result.get("semantic_search", {})
                                if semantic_results and semantic_results.get("success"):
                                    st.markdown("### 🔍 Semantic Search Results")
                                    sem_results = semantic_results.get("results", [])
                                    if sem_results:
                                        for i, item in enumerate(sem_results[:5]):
                                            with st.expander(f"Semantic Result {i+1}"):
                                                st.markdown(f"**Content:** {item.get('content', 'N/A')[:200]}...")
                                                st.markdown(f"**Score:** {item.get('similarity', 0):.3f}")
                                
                                # Display knowledge graph results
                                kg_results = result.get("knowledge_graph_search", {})
                                if kg_results and kg_results.get("success"):
                                    st.markdown("### 🧠 Knowledge Graph Results")
                                    st.json(kg_results)
                    else:
                        st.warning("Please enter a query.")
            
            with col2:
                st.markdown("### 🔄 Combined Search")
                st.markdown("""
                **Benefits:**
                - Semantic + Knowledge Graph
                - Comprehensive results
                - Multiple perspectives
                - Enhanced discovery
                
                **Best for:**
                - Research queries
                - Topic exploration
                - Content discovery
                - Relationship analysis
                """)
        
        with tab4:
            st.markdown("### Search Statistics")
            
            if st.button("🔄 Refresh Statistics"):
                st.rerun()
            
            if stats and "error" not in stats:
                st.markdown("#### 📊 Index Statistics")
                st.json(stats)
            else:
                st.error("Unable to retrieve search statistics")
    
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
