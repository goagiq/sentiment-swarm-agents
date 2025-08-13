#!/usr/bin/env python3
"""
Simple script to process all PDFs in the data directory using existing main.py functionality.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import after path modification
from main import OptimizedMCPServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def process_all_pdfs():
    """Process all PDFs in the data directory using the MCP server."""
    
    # Initialize the MCP server
    mcp_server = OptimizedMCPServer()
    
    # Find all PDF files
    data_dir = Path("data")
    pdf_files = list(data_dir.glob("*.pdf"))
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    processed_files = []
    failed_files = []
    
    # Process each PDF file
    for pdf_file in pdf_files:
        try:
            logger.info(f"Processing: {pdf_file}")
            
            # Use the existing process_pdf_enhanced_multilingual function
            result = await mcp_server.process_pdf_enhanced_multilingual(
                pdf_path=str(pdf_file),
                language="auto",
                generate_report=True,
                output_path=None
            )
            
            if result and result.get("status") == "success":
                processed_files.append(str(pdf_file))
                logger.info(f"Successfully processed: {pdf_file}")
            else:
                failed_files.append(str(pdf_file))
                logger.error(f"Failed to process: {pdf_file}")
                
        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {str(e)}")
            failed_files.append(str(pdf_file))
    
    # Generate comprehensive graph report
    logger.info("Generating comprehensive graph report...")
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"Results/comprehensive_pdf_graph_report_{timestamp}.html"
        
        report_result = await mcp_server.generate_graph_report(
            output_path=report_path,
            target_language="en"
        )
        
        if report_result:
            logger.info(f"Graph report generated: {report_path}")
        else:
            logger.error("Failed to generate graph report")
            
    except Exception as e:
        logger.error(f"Error generating graph report: {str(e)}")
    
    # Print summary
    print("\n" + "="*60)
    print("PDF PROCESSING SUMMARY")
    print("="*60)
    print(f"Total files: {len(pdf_files)}")
    print(f"Successfully processed: {len(processed_files)}")
    print(f"Failed: {len(failed_files)}")
    print(f"Success rate: {len(processed_files)/len(pdf_files)*100:.1f}%")
    
    if processed_files:
        print("\nSuccessfully processed files:")
        for file in processed_files:
            print(f"  ✓ {file}")
    
    if failed_files:
        print("\nFailed files:")
        for file in failed_files:
            print(f"  ✗ {file}")


if __name__ == "__main__":
    asyncio.run(process_all_pdfs())
