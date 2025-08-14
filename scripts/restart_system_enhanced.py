#!/usr/bin/env python3
"""
Enhanced System Restart Script for Sentiment Analysis System
Checks for running processes and ports, kills them, and waits before restarting.
"""

import os
import sys
import time
import subprocess
import psutil
import signal
import requests
from pathlib import Path

class EnhancedSystemRestartManager:
    """Enhanced system restart manager with port checking and process management."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.main_script = self.project_root / "main.py"
        self.ports_to_check = [8003, 8004]  # Main server and test ports
        self.python_processes = []
        
    def check_port_in_use(self, port):
        """Check if a port is in use."""
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', port))
                return result == 0
        except Exception:
            return False
    
    def find_python_processes(self):
        """Find all Python processes related to this project."""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    cmdline = proc.info['cmdline']
                    if cmdline and any('main.py' in str(arg) for arg in cmdline):
                        processes.append(proc)
                    elif cmdline and any('sentiment' in str(arg).lower() for arg in cmdline):
                        processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return processes
    
    def kill_process(self, proc):
        """Kill a process gracefully, then forcefully if needed."""
        try:
            print(f"   Killing process {proc.pid} ({proc.name()})")
            proc.terminate()
            try:
                proc.wait(timeout=5)
                print(f"   ✅ Process {proc.pid} terminated gracefully")
            except psutil.TimeoutExpired:
                print(f"   ⚠️ Process {proc.pid} didn't terminate, killing forcefully")
                proc.kill()
                proc.wait()
                print(f"   ✅ Process {proc.pid} killed forcefully")
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            print(f"   ⚠️ Could not kill process {proc.pid}: {e}")
    
    def kill_all_python_processes(self):
        """Kill all Python processes related to this project."""
        print("🔍 Finding Python processes...")
        processes = self.find_python_processes()
        
        if not processes:
            print("✅ No Python processes found")
            return
        
        print(f"🔍 Found {len(processes)} Python processes:")
        for proc in processes:
            try:
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else 'N/A'
                print(f"   PID {proc.pid}: {cmdline}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        print("\n🛑 Killing Python processes...")
        for proc in processes:
            self.kill_process(proc)
    
    def check_ports(self):
        """Check if any of our ports are in use."""
        print("🔍 Checking ports...")
        ports_in_use = []
        for port in self.ports_to_check:
            if self.check_port_in_use(port):
                ports_in_use.append(port)
                print(f"   ⚠️ Port {port} is in use")
            else:
                print(f"   ✅ Port {port} is free")
        return ports_in_use
    
    def wait_for_ports_to_free(self, timeout=60):
        """Wait for ports to become free."""
        print(f"⏳ Waiting for ports to become free (timeout: {timeout}s)...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            ports_in_use = self.check_ports()
            if not ports_in_use:
                print("✅ All ports are now free")
                return True
            
            print(f"   Still waiting... ({int(timeout - (time.time() - start_time))}s remaining)")
            time.sleep(2)
        
        print(f"⚠️ Timeout reached, some ports may still be in use")
        return False
    
    def start_server(self):
        """Start the main server."""
        if not self.main_script.exists():
            print(f"❌ Main script not found: {self.main_script}")
            return False
        
        print(f"🚀 Starting server: {self.main_script}")
        try:
            # Start the server in the background
            process = subprocess.Popen(
                [sys.executable, str(self.main_script)],
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            print(f"✅ Server started with PID: {process.pid}")
            return process
            
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return None
    
    def wait_for_server_ready(self, timeout=60):
        """Wait for the server to be ready."""
        print(f"⏳ Waiting for server to be ready (timeout: {timeout}s)...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get("http://localhost:8003/health", timeout=5)
                if response.status_code == 200:
                    print("✅ Server is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            remaining = int(timeout - (time.time() - start_time))
            print(f"   Still waiting... ({remaining}s remaining)")
            time.sleep(2)
        
        print("⚠️ Timeout reached, server may not be ready")
        return False
    
    def test_mcp_endpoints(self):
        """Test MCP endpoints to verify they're working."""
        print("🔧 Testing MCP endpoints...")
        
        # Test health endpoint
        try:
            response = requests.get("http://localhost:8003/mcp-health", timeout=10)
            print(f"✅ MCP Health: {response.status_code}")
            if response.status_code == 200:
                print(f"   Response: {response.json()}")
        except Exception as e:
            print(f"❌ MCP Health Error: {e}")
        
        # Test MCP initialization
        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
            
            mcp_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "test-client", "version": "1.0.0"}
                }
            }
            
            response = requests.post(
                "http://localhost:8003/mcp/",
                json=mcp_request,
                headers=headers,
                timeout=10
            )
            print(f"✅ MCP Initialize: {response.status_code}")
            if response.status_code == 200:
                print("🎉 MCP server is working correctly!")
            else:
                print(f"   Response: {response.text[:200]}...")
                
        except Exception as e:
            print(f"❌ MCP Initialize Error: {e}")
    
    def quick_restart(self):
        """Quick restart: kill processes and restart server."""
        print("🚀 Quick Restart")
        print("=" * 50)
        
        # Kill processes
        self.kill_all_python_processes()
        
        # Wait for ports to free
        self.wait_for_ports_to_free()
        
        # Start server
        process = self.start_server()
        if process:
            print("✅ Quick restart completed")
            return process
        else:
            print("❌ Quick restart failed")
            return None
    
    def full_restart(self):
        """Full restart: kill processes, wait, restart, and verify."""
        print("🚀 Full Restart with Verification")
        print("=" * 50)
        
        # Kill processes
        self.kill_all_python_processes()
        
        # Wait for ports to free
        if not self.wait_for_ports_to_free():
            print("⚠️ Some ports may still be in use, continuing anyway...")
        
        # Wait additional time for cleanup
        print("⏳ Waiting 30 seconds for system cleanup...")
        time.sleep(30)
        
        # Start server
        process = self.start_server()
        if not process:
            print("❌ Failed to start server")
            return None
        
        # Wait for server to be ready
        if not self.wait_for_server_ready():
            print("⚠️ Server may not be fully ready")
        
        # Test MCP endpoints
        self.test_mcp_endpoints()
        
        print("✅ Full restart completed")
        return process
    
    def kill_only(self):
        """Only kill processes, don't restart."""
        print("🛑 Kill Processes Only")
        print("=" * 50)
        
        self.kill_all_python_processes()
        self.wait_for_ports_to_free()
        print("✅ All processes killed")

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python restart_system_enhanced.py [quick|full|kill]")
        print("  quick: Kill processes and restart server")
        print("  full:  Kill processes, wait 30s, restart, and verify")
        print("  kill:  Only kill processes, don't restart")
        return
    
    command = sys.argv[1].lower()
    manager = EnhancedSystemRestartManager()
    
    try:
        if command == "quick":
            manager.quick_restart()
        elif command == "full":
            manager.full_restart()
        elif command == "kill":
            manager.kill_only()
        else:
            print(f"❌ Unknown command: {command}")
            print("Use: quick, full, or kill")
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

