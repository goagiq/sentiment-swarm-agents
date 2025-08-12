#!/usr/bin/env python3
"""
Check processing status and generate report for Classical Chinese PDF.
"""

import asyncio
import os
import sys
import json
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.vector_db import vector_db
from src.agents.knowledge_graph_agent import KnowledgeGraphAgent


async def check_processing_status():
    """Check the processing status and generate a report."""
    print("🔍 Checking Processing Status for Classical Chinese PDF")
    print("=" * 60)
    
    # Check vector database
    print("\n📊 Vector Database Status:")
    try:
        stats = vector_db.get_database_stats()
        print(f"✅ Total documents: {stats['total_documents']}")
        for collection, info in stats['collections'].items():
            print(f"   └─ {collection}: {info['document_count']} documents")
    except Exception as e:
        print(f"❌ Error checking vector database: {e}")
    
    # Check knowledge graph
    print("\n🧠 Knowledge Graph Status:")
    try:
        agent = KnowledgeGraphAgent()
        graph_stats = agent._get_graph_stats()
        print(f"✅ Nodes: {graph_stats.get('nodes', 0)}")
        print(f"✅ Edges: {graph_stats.get('edges', 0)}")
        print(f"✅ Languages: {graph_stats.get('languages', [])}")
    except Exception as e:
        print(f"❌ Error checking knowledge graph: {e}")
    
    # Check for recent reports
    print("\n📄 Recent Reports:")
    try:
        reports_dir = "Results/reports"
        if os.path.exists(reports_dir):
            files = os.listdir(reports_dir)
            if files:
                for file in sorted(files, reverse=True)[:5]:
                    print(f"   └─ {file}")
            else:
                print("   └─ No reports found")
        else:
            print("   └─ Reports directory not found")
    except Exception as e:
        print(f"❌ Error checking reports: {e}")
    
    # Generate a simple report
    print("\n📋 Generating Status Report...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"Results/processing_status_{timestamp}.json"
    
    os.makedirs("Results", exist_ok=True)
    
    report = {
        "timestamp": timestamp,
        "pdf_file": "data/Classical Chinese Sample 22208_0_8.pdf",
        "vector_db": stats,
        "knowledge_graph": graph_stats if 'graph_stats' in locals() else {"error": "Could not retrieve"},
        "status": "completed" if stats['total_documents'] > 0 else "failed"
    }
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Status report saved to: {report_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Processing Summary:")
    print(f"✅ Vector Database: {stats['total_documents']} documents stored")
    if 'graph_stats' in locals():
        print(f"✅ Knowledge Graph: {graph_stats.get('nodes', 0)} nodes, {graph_stats.get('edges', 0)} edges")
    print("✅ Status Report: Generated")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(check_processing_status())
