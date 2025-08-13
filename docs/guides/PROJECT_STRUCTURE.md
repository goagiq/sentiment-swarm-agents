# Project Structure

## 📁 Root Directory
```
Sentiment/
├── main.py                          # Main entry point (Phases 1-5 complete)
├── CONTENT_ANALYSIS_IMPLEMENTATION_PLAN.md  # Project blueprint
├── README.md                        # Main project documentation
├── requirements.prod.txt            # Production dependencies
├── pyproject.toml                   # Project configuration
├── .gitignore                       # Git ignore rules
├── .dockerignore                    # Docker ignore rules
├── Dockerfile                       # Docker configuration
├── docker-compose.prod.yml          # Production Docker Compose
├── env.example                      # Environment variables template
├── env.production                   # Production environment
├── uv.lock                          # Dependency lock file
└── PROJECT_STRUCTURE.md             # This file
```

## 📁 Core Directories

### `/src/` - Source Code
```
src/
├── __init__.py
├── pytest.ini                      # Test configuration
├── api/                            # FastAPI endpoints
├── agents/                         # All agent implementations
├── core/                           # Core utilities and services
├── config/                         # Configuration management
├── mcp_servers/                    # MCP server implementations
└── archive/                        # Archived/old code
```

### `/Test/` - Test Files
```
Test/
├── test_phase1.py                  # Phase 1 tests
├── test_phase2.py                  # Phase 2 tests
├── test_phase3.py                  # Phase 3 tests
├── test_phase4.py                  # Phase 4 tests
├── test_phase5.py                  # Phase 5 tests
├── run_all_tests.py                # Test runner
├── archive/                        # Old test files
└── *.json                          # Test results
```

### `/ui/` - User Interface
```
ui/
├── main.py                         # Main Streamlit UI
├── landing_page.py                 # Landing page
├── README.md                       # UI documentation
└── UI_README.md                    # UI implementation guide
```

### `/docs/` - Documentation
```
docs/
├── PHASE1_IMPLEMENTATION_SUMMARY.md
├── PHASE4_IMPLEMENTATION_SUMMARY.md
├── PERFORMANCE_OPTIMIZATION_PLAN.md
├── PERFORMANCE_OPTIMIZATION_RESULTS.md
├── PROJECT_DESIGN_FRAMEWORK.md
└── PROJECT_CLEANUP_SUMMARY.md
```

### `/cache/` - Cache and Data
```
cache/
├── chroma_db/                      # Vector database
└── ...                             # Other cache files
```

### `/Results/` - Generated Results
```
Results/
├── reports/                        # Generated reports
├── exports/                        # Data exports
└── ...                             # Other results
```

## 🚀 Quick Start

1. **Install dependencies**: `pip install -r requirements.prod.txt`
2. **Run the system**: `python main.py`
3. **Access services**:
   - Main UI: http://localhost:8501
   - Landing Page: http://localhost:8502
   - API Docs: http://localhost:8003/docs
   - MCP Server: http://localhost:8000/mcp

## 📋 Implementation Status

- ✅ **Phase 1**: Core Sentiment Analysis
- ✅ **Phase 2**: Business Intelligence
- ✅ **Phase 3**: Advanced Analytics
- ✅ **Phase 4**: Export & Automation
- ✅ **Phase 5**: Semantic Search & Reflection

## 🧹 Cleanup Summary

- Moved test files to `/Test/` directory
- Moved documentation to `/docs/` directory
- Moved UI docs to `/ui/` directory
- Removed old main files to `/src/archive/`
- Cleaned up cache and temporary directories
- Removed duplicate files and old scripts
- Organized project following @DesignFramework guidelines
