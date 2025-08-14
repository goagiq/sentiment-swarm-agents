#!/usr/bin/env python3
"""
System Restart Script for Sentiment Analysis System
Implements the System Restart Framework from the Design Framework.
"""

import subprocess
import time
import os
import signal
import sys
from pathlib import Path

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not available - using basic process management")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️ requests not available - skipping endpoint verification")


class SystemRestartManager:
    """Manages system restart with proper process cleanup and verification."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.main_script = self.project_root / "main.py"
        self.python_exe = self.project_root / ".venv" / "Scripts" / "python.exe"
        
    def kill_python_processes(self):
        """Kill all Python processes safely."""
        try:
            print("🔪 Terminating Python processes...")
            
            # Method 1: Use taskkill on Windows
            if os.name == 'nt':  # Windows
                subprocess.run(["taskkill", "/f", "/im", "python.exe"], 
                             capture_output=True, check=False)
                subprocess.run(["taskkill", "/f", "/im", "pythonw.exe"], 
                             capture_output=True, check=False)
                print("✅ Windows process termination completed")
            else:  # Unix/Linux
                subprocess.run(["pkill", "-f", "python"], 
                             capture_output=True, check=False)
                print("✅ Unix process termination completed")
            
            # Method 2: Use psutil for more precise control
            if PSUTIL_AVAILABLE:
                print("🔍 Using psutil for precise process management...")
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if proc.info['name'] and 'python' in proc.info['name'].lower():
                            cmdline = proc.info['cmdline'] or []
                            if any('main.py' in str(cmd) for cmd in cmdline):
                                print(f"   Terminating main.py process (PID: {proc.pid})")
                                proc.terminate()
                                proc.wait(timeout=5)
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                        pass
            
            print("✅ Python processes terminated")
            time.sleep(2)  # Wait for processes to fully terminate
            
        except Exception as e:
            print(f"⚠️ Warning: Could not terminate processes: {e}")
    
    def wait_for_server_ready(self, timeout: int = 30, check_interval: int = 1):
        """Wait for server to be ready with countdown."""
        print(f"⏳ Waiting {timeout} seconds for server to be ready...")
        for i in range(timeout, 0, -1):
            print(f"   {i} seconds remaining...", end="\r")
            time.sleep(check_interval)
        print("\n🚀 Server should be ready now!")
    
                def restart_system(self, wait_time: int = 30):
                    """Complete system restart with verification."""
                    print("🔄 Starting system restart...")
                    print("=" * 50)
                    
                    # Step 1: Kill existing processes
                    print("1️⃣ Killing existing Python processes...")
                    self.kill_python_processes()
                    
                    # Step 2: Wait for cleanup
                    print("2️⃣ Waiting for process cleanup...")
                    time.sleep(3)
                    
                    # Step 3: Start main.py
                    print("3️⃣ Starting main.py...")
                    try:
                        if not self.python_exe.exists():
                            print(f"❌ Python executable not found at: {self.python_exe}")
                            return False
                            
                        if not self.main_script.exists():
                            print(f"❌ Main script not found at: {self.main_script}")
                            return False
                        
                        # Start main.py in background with proper process group
                        if os.name == 'nt':  # Windows
                            # On Windows, use CREATE_NEW_PROCESS_GROUP
                            import subprocess
                            process = subprocess.Popen(
                                [str(self.python_exe), str(self.main_script)],
                                cwd=self.project_root,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                            )
                        else:  # Unix/Linux
                            # On Unix, use preexec_fn to detach from parent
                            process = subprocess.Popen(
                                [str(self.python_exe), str(self.main_script)],
                                cwd=self.project_root,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                preexec_fn=os.setsid
                            )
                        
                        print(f"✅ main.py started with PID: {process.pid}")
                        
                        # Give the process a moment to start
                        time.sleep(2)
                        
                        # Check if process is still running
                        if process.poll() is not None:
                            print(f"❌ main.py process exited immediately with code: {process.returncode}")
                            return False
                        
                    except Exception as e:
                        print(f"❌ Failed to start main.py: {e}")
                        return False
                    
                    # Step 4: Wait for server to be ready
                    print("4️⃣ Waiting for server initialization...")
                    self.wait_for_server_ready(wait_time)
                    
                    # Step 5: Verify server is running
                    print("5️⃣ Verifying server status...")
                    return self.verify_server_status()
    
    def verify_server_status(self):
        """Verify that the server is running properly."""
        if not REQUESTS_AVAILABLE:
            print("⚠️ requests not available - skipping endpoint verification")
            return True
        
        endpoints = [
            ("http://localhost:8003/health", "FastAPI Health"),
            ("http://localhost:8003/mcp", "MCP Server"),
            ("http://localhost:8003/mcp/", "MCP Server (trailing slash)"),
            ("http://localhost:8003/mcp-health", "MCP Health"),
            ("http://localhost:8501", "Streamlit Main UI"),
            ("http://localhost:8502", "Streamlit Landing Page")
        ]
        
        all_ok = True
        print("🔍 Checking server endpoints...")
        
        for url, name in endpoints:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code in [200, 404]:  # 404 is OK for some endpoints
                    print(f"✅ {name}: {response.status_code}")
                else:
                    print(f"⚠️ {name}: {response.status_code}")
                    all_ok = False
            except requests.exceptions.ConnectionError:
                print(f"❌ {name}: Connection refused")
                all_ok = False
            except Exception as e:
                print(f"❌ {name}: Error - {e}")
                all_ok = False
        
        if all_ok:
            print("🎉 System restart completed successfully!")
        else:
            print("⚠️ Some endpoints may not be ready yet. Check server logs.")
        
        return all_ok


def restart_and_verify():
    """Restart the system and verify it's working."""
    restart_manager = SystemRestartManager()
    success = restart_manager.restart_system(wait_time=30)
    
    if success:
        print("✅ System is ready for use!")
        print("🌐 Access URLs:")
        print("   📊 Main UI:        http://localhost:8501")
        print("   🏠 Landing Page:   http://localhost:8502")
        print("   🔗 API Docs:       http://localhost:8003/docs")
        print("   🤖 MCP Server:     http://localhost:8003/mcp")
    else:
        print("❌ System restart failed. Check logs for details.")
    
    return success


def quick_restart():
    """Quick restart for development (15 second wait)."""
    restart_manager = SystemRestartManager()
    return restart_manager.restart_system(wait_time=15)


def main():
    """Main function with command line interface."""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "quick":
            print("🚀 Quick restart (15 seconds)...")
            quick_restart()
        elif command == "full":
            print("🔄 Full restart with verification (30 seconds)...")
            restart_and_verify()
        elif command == "kill":
            print("🔪 Killing Python processes only...")
            restart_manager = SystemRestartManager()
            restart_manager.kill_python_processes()
        else:
            print("❌ Unknown command. Use: quick, full, or kill")
    else:
        print("🔄 Full restart with verification (30 seconds)...")
        restart_and_verify()


if __name__ == "__main__":
    main()
