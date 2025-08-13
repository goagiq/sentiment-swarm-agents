#!/usr/bin/env python3
"""
Script to stop running Python processes and restart the application cleanly.
"""

import subprocess
import time
import os
from pathlib import Path


def stop_python_processes():
    """Stop all running Python processes."""
    print("🛑 Stopping Python processes...")
    
    try:
        # Get list of Python processes
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
            capture_output=True, text=True, shell=True
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            if lines and lines[0]:
                print(f"Found {len(lines)} Python processes")
                
                # Stop each process
                for line in lines:
                    if line.strip():
                        parts = line.split(',')
                        if len(parts) >= 2:
                            pid = parts[1].strip('"')
                            try:
                                subprocess.run(["taskkill", "/PID", pid, "/F"], 
                                             capture_output=True, shell=True)
                                print(f"Stopped process PID: {pid}")
                            except Exception as e:
                                print(f"Could not stop PID {pid}: {e}")
            else:
                print("No Python processes found")
        else:
            print("Could not get process list")
            
    except Exception as e:
        print(f"Error stopping processes: {e}")


def wait_for_cleanup():
    """Wait for file handles to be released."""
    print("⏳ Waiting for file handles to be released...")
    time.sleep(3)


def clear_chromadb():
    """Clear ChromaDB after processes are stopped."""
    try:
        chroma_db_path = Path("cache/chroma_db")
        if chroma_db_path.exists():
            import shutil
            shutil.rmtree(chroma_db_path)
            chroma_db_path.mkdir(parents=True, exist_ok=True)
            print("✅ ChromaDB cleared successfully")
        else:
            print("ChromaDB directory not found")
    except Exception as e:
        print(f"Error clearing ChromaDB: {e}")


def start_application():
    """Start the application."""
    print("\n🚀 Starting application...")
    print("Choose an option:")
    print("1. Start main application (python main.py)")
    print("2. Start MCP server (python -m src.mcp.server)")
    print("3. Start both")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ")
    
    if choice == "1":
        print("Starting main application...")
        subprocess.Popen(["python", "main.py"])
    elif choice == "2":
        print("Starting MCP server...")
        subprocess.Popen(["python", "-m", "src.mcp.server"])
    elif choice == "3":
        print("Starting both applications...")
        subprocess.Popen(["python", "main.py"])
        time.sleep(2)
        subprocess.Popen(["python", "-m", "src.mcp.server"])
    elif choice == "4":
        print("Exiting...")
        return
    else:
        print("Invalid choice. Exiting...")


def main():
    """Main function."""
    print("🔄 Clean Restart Utility")
    print("=" * 50)
    print("This will:")
    print("1. Stop all running Python processes")
    print("2. Clear ChromaDB completely")
    print("3. Restart the application fresh")
    print()
    
    response = input("Proceed with clean restart? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        # Stop processes
        stop_python_processes()
        
        # Wait for cleanup
        wait_for_cleanup()
        
        # Clear ChromaDB
        clear_chromadb()
        
        # Start application
        start_application()
        
        print("\n✅ Clean restart completed!")
    else:
        print("Restart cancelled.")


if __name__ == "__main__":
    main()
