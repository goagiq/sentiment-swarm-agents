#!/usr/bin/env python3
"""
Step 9: Documentation and Cleanup Testing
Validates documentation updates and identifies files for cleanup
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class DocumentationCleanupTester:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results = {}
        self.cleanup_files = []
        self.documentation_updates = []
        
    def log_test(self, category: str, test_name: str, status: str, 
                 details: str = "", files: List[str] = None):
        """Log test results"""
        if category not in self.results:
            self.results[category] = {}
            
        self.results[category][test_name] = {
            "status": status,
            "details": details,
            "files": files or []
        }
    
    def test_documentation_structure(self):
        """Test documentation structure and completeness"""
        print("📚 Testing Documentation Structure...")
        
        # Check main documentation files
        doc_files = [
            "README.md",
            "docs/MCP_SERVER_OPTIMIZATION_PLAN.md",
            "docs/CONFIGURABLE_MODELS_GUIDE.md",
            "docs/UNIFIED_AGENTS_GUIDE.md"
        ]
        
        missing_docs = []
        existing_docs = []
        
        for doc_file in doc_files:
            doc_path = self.project_root / doc_file
            if doc_path.exists():
                existing_docs.append(doc_file)
            else:
                missing_docs.append(doc_file)
        
        if missing_docs:
            self.log_test("Documentation", "structure", "FAIL", 
                         f"Missing documentation files: {missing_docs}", missing_docs)
            print(f"  ❌ Missing documentation: {missing_docs}")
        else:
            self.log_test("Documentation", "structure", "PASS", 
                         f"All documentation files present: {existing_docs}", existing_docs)
            print(f"  ✅ Documentation structure: {len(existing_docs)} files found")
    
    def test_readme_updates(self):
        """Test README.md for consolidated server references"""
        print("\n📖 Testing README Updates...")
        
        readme_path = self.project_root / "README.md"
        if not readme_path.exists():
            self.log_test("README", "updates", "FAIL", "README.md not found")
            print("  ❌ README.md not found")
            return
        
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for consolidated server references
            consolidated_keywords = [
                "Consolidated MCP Server",
                "consolidated_mcp_server",
                "PDF Processing Server",
                "Audio Processing Server", 
                "Video Processing Server",
                "Website Processing Server"
            ]
            
            found_keywords = []
            missing_keywords = []
            
            for keyword in consolidated_keywords:
                if keyword in content:
                    found_keywords.append(keyword)
                else:
                    missing_keywords.append(keyword)
            
            if missing_keywords:
                self.log_test("README", "updates", "PARTIAL", 
                             f"Missing keywords: {missing_keywords}", missing_keywords)
                print(f"  ⚠️  README partially updated: {len(found_keywords)}/{len(consolidated_keywords)} keywords found")
            else:
                self.log_test("README", "updates", "PASS", 
                             f"All consolidated server keywords found: {found_keywords}", found_keywords)
                print(f"  ✅ README fully updated: {len(found_keywords)} keywords found")
                
        except Exception as e:
            self.log_test("README", "updates", "FAIL", f"Error reading README: {e}")
            print(f"  ❌ README read error: {e}")
    
    def test_examples_cleanup(self):
        """Test examples directory for old individual MCP server files"""
        print("\n📁 Testing Examples Cleanup...")
        
        examples_dir = self.project_root / "examples"
        if not examples_dir.exists():
            self.log_test("Examples", "cleanup", "SKIP", "Examples directory not found")
            print("  ⏭️  Examples directory not found")
            return
        
        # Look for old individual MCP server examples
        old_examples = []
        consolidated_examples = []
        
        for example_file in examples_dir.glob("*.py"):
            filename = example_file.name
            if any(keyword in filename.lower() for keyword in 
                   ["text_agent", "audio_agent", "vision_agent", "web_agent"]):
                old_examples.append(filename)
            elif "consolidated" in filename.lower() or "mcp" in filename.lower():
                consolidated_examples.append(filename)
        
        if old_examples:
            self.log_test("Examples", "cleanup", "NEEDS_CLEANUP", 
                         f"Old examples found: {old_examples}", old_examples)
            print(f"  ⚠️  Old examples need cleanup: {old_examples}")
            self.cleanup_files.extend([f"examples/{f}" for f in old_examples])
        else:
            self.log_test("Examples", "cleanup", "PASS", 
                         f"Examples directory clean, consolidated examples: {consolidated_examples}", 
                         consolidated_examples)
            print(f"  ✅ Examples directory clean: {len(consolidated_examples)} consolidated examples")
    
    def test_docs_cleanup(self):
        """Test docs directory for old individual MCP server documentation"""
        print("\n📄 Testing Docs Cleanup...")
        
        docs_dir = self.project_root / "docs"
        if not docs_dir.exists():
            self.log_test("Docs", "cleanup", "SKIP", "Docs directory not found")
            print("  ⏭️  Docs directory not found")
            return
        
        # Look for old individual MCP server documentation
        old_docs = []
        consolidated_docs = []
        
        for doc_file in docs_dir.glob("*.md"):
            filename = doc_file.name
            if any(keyword in filename.lower() for keyword in 
                   ["text_agent", "audio_agent", "vision_agent", "web_agent"]):
                old_docs.append(filename)
            elif "consolidated" in filename.lower() or "optimization" in filename.lower():
                consolidated_docs.append(filename)
        
        if old_docs:
            self.log_test("Docs", "cleanup", "NEEDS_CLEANUP", 
                         f"Old documentation found: {old_docs}", old_docs)
            print(f"  ⚠️  Old docs need cleanup: {old_docs}")
            self.cleanup_files.extend([f"docs/{f}" for f in old_docs])
        else:
            self.log_test("Docs", "cleanup", "PASS", 
                         f"Docs directory clean, consolidated docs: {consolidated_docs}", 
                         consolidated_docs)
            print(f"  ✅ Docs directory clean: {len(consolidated_docs)} consolidated docs")
    
    def test_old_mcp_files(self):
        """Test for old individual MCP server files in src/mcp"""
        print("\n🔧 Testing Old MCP Files...")
        
        mcp_dir = self.project_root / "src" / "mcp"
        if not mcp_dir.exists():
            self.log_test("MCP", "old_files", "SKIP", "MCP directory not found")
            print("  ⏭️  MCP directory not found")
            return
        
        # Look for old individual MCP server files
        old_files = []
        consolidated_files = []
        
        for mcp_file in mcp_dir.glob("*.py"):
            filename = mcp_file.name
            if any(keyword in filename.lower() for keyword in 
                   ["text_agent", "audio_agent", "vision_agent", "web_agent"]):
                old_files.append(filename)
            elif "consolidated" in filename.lower() or "processing_server" in filename.lower():
                consolidated_files.append(filename)
        
        if old_files:
            self.log_test("MCP", "old_files", "NEEDS_CLEANUP", 
                         f"Old MCP files found: {old_files}", old_files)
            print(f"  ⚠️  Old MCP files need cleanup: {old_files}")
            self.cleanup_files.extend([f"src/mcp/{f}" for f in old_files])
        else:
            self.log_test("MCP", "old_files", "PASS", 
                         f"MCP directory clean, consolidated files: {consolidated_files}", 
                         consolidated_files)
            print(f"  ✅ MCP directory clean: {len(consolidated_files)} consolidated files")
    
    def test_configuration_documentation(self):
        """Test configuration documentation completeness"""
        print("\n⚙️  Testing Configuration Documentation...")
        
        config_docs = [
            "docs/CONFIGURABLE_MODELS_GUIDE.md",
            "docs/OLLAMA_CONFIGURATION_GUIDE.md"
        ]
        
        missing_config_docs = []
        existing_config_docs = []
        
        for doc_file in config_docs:
            doc_path = self.project_root / doc_file
            if doc_path.exists():
                existing_config_docs.append(doc_file)
            else:
                missing_config_docs.append(doc_file)
        
        if missing_config_docs:
            self.log_test("Config", "documentation", "PARTIAL", 
                         f"Missing config docs: {missing_config_docs}", missing_config_docs)
            print(f"  ⚠️  Configuration docs incomplete: {missing_config_docs}")
        else:
            self.log_test("Config", "documentation", "PASS", 
                         f"All config docs present: {existing_config_docs}", existing_config_docs)
            print(f"  ✅ Configuration documentation complete: {len(existing_config_docs)} files")
    
    def test_test_files_cleanup(self):
        """Test test directory for old individual MCP server tests"""
        print("\n🧪 Testing Test Files Cleanup...")
        
        test_dir = self.project_root / "Test"
        if not test_dir.exists():
            self.log_test("Tests", "cleanup", "SKIP", "Test directory not found")
            print("  ⏭️  Test directory not found")
            return
        
        # Look for old individual MCP server tests
        old_tests = []
        consolidated_tests = []
        
        for test_file in test_dir.glob("*.py"):
            filename = test_file.name
            if any(keyword in filename.lower() for keyword in 
                   ["text_agent", "audio_agent", "vision_agent", "web_agent"]):
                old_tests.append(filename)
            elif any(keyword in filename.lower() for keyword in 
                     ["consolidated", "step", "performance", "configuration"]):
                consolidated_tests.append(filename)
        
        if old_tests:
            self.log_test("Tests", "cleanup", "NEEDS_CLEANUP", 
                         f"Old test files found: {old_tests}", old_tests)
            print(f"  ⚠️  Old tests need cleanup: {old_tests}")
            self.cleanup_files.extend([f"Test/{f}" for f in old_tests])
        else:
            self.log_test("Tests", "cleanup", "PASS", 
                         f"Test directory clean, consolidated tests: {consolidated_tests}", 
                         consolidated_tests)
            print(f"  ✅ Test directory clean: {len(consolidated_tests)} consolidated tests")
    
    def generate_cleanup_report(self):
        """Generate cleanup recommendations"""
        print("\n🗑️  Generating Cleanup Report...")
        
        if self.cleanup_files:
            cleanup_report = {
                "files_to_remove": self.cleanup_files,
                "reason": "Old individual MCP server files replaced by consolidated servers",
                "backup_recommendation": "Consider backing up before removal"
            }
            
            self.log_test("Cleanup", "report", "NEEDS_ACTION", 
                         f"{len(self.cleanup_files)} files need cleanup", self.cleanup_files)
            print(f"  ⚠️  {len(self.cleanup_files)} files need cleanup")
            
            # Save cleanup report
            cleanup_file = self.project_root / "Test" / "step9_cleanup_report.json"
            with open(cleanup_file, 'w', encoding='utf-8') as f:
                json.dump(cleanup_report, f, indent=2)
            
            print(f"  💾 Cleanup report saved to: {cleanup_file}")
        else:
            self.log_test("Cleanup", "report", "PASS", "No files need cleanup")
            print("  ✅ No files need cleanup")
    
    def run_all_tests(self):
        """Run all documentation and cleanup tests"""
        print("🚀 Starting Step 9: Documentation and Cleanup Testing")
        print("=" * 60)
        
        # Run all test categories
        self.test_documentation_structure()
        self.test_readme_updates()
        self.test_examples_cleanup()
        self.test_docs_cleanup()
        self.test_old_mcp_files()
        self.test_configuration_documentation()
        self.test_test_files_cleanup()
        self.generate_cleanup_report()
        
        # Generate summary report
        self.generate_report()
    
    def generate_report(self):
        """Generate documentation and cleanup test report"""
        print("\n" + "=" * 60)
        print("📊 STEP 9 DOCUMENTATION AND CLEANUP SUMMARY")
        print("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        partial_tests = 0
        skipped_tests = 0
        needs_cleanup = 0
        
        for category, tests in self.results.items():
            print(f"\n📁 {category.upper()} CATEGORY:")
            for test_name, test_result in tests.items():
                total_tests += 1
                status = test_result["status"]
                
                if status == "PASS":
                    passed_tests += 1
                    status_icon = "✅"
                elif status == "FAIL":
                    failed_tests += 1
                    status_icon = "❌"
                elif status == "PARTIAL":
                    partial_tests += 1
                    status_icon = "⚠️"
                elif status == "SKIP":
                    skipped_tests += 1
                    status_icon = "⏭️"
                elif status == "NEEDS_CLEANUP":
                    needs_cleanup += 1
                    status_icon = "🗑️"
                else:
                    status_icon = "❓"
                
                print(f"  {status_icon} {test_name}: {test_result['details']}")
                if test_result['files']:
                    print(f"    Files: {', '.join(test_result['files'])}")
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"  Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        print(f"  Partial: {partial_tests} ({partial_tests/total_tests*100:.1f}%)")
        print(f"  Skipped: {skipped_tests} ({skipped_tests/total_tests*100:.1f}%)")
        print(f"  Needs Cleanup: {needs_cleanup} ({needs_cleanup/total_tests*100:.1f}%)")
        
        # Save detailed results
        results_file = self.project_root / "Test" / "step9_documentation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "step": 9,
                "timestamp": __import__('time').time(),
                "statistics": {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": failed_tests,
                    "partial_tests": partial_tests,
                    "skipped_tests": skipped_tests,
                    "needs_cleanup": needs_cleanup
                },
                "results": self.results,
                "cleanup_files": self.cleanup_files
            }, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        if failed_tests == 0 and needs_cleanup == 0:
            print("\n🎉 Step 9 completed successfully!")
            print("✅ All documentation and cleanup tests passed!")
        else:
            print(f"\n⚠️  {failed_tests} tests failed, {needs_cleanup} need cleanup.")


def main():
    """Main test execution"""
    try:
        tester = DocumentationCleanupTester()
        tester.run_all_tests()
    except Exception as e:
        print(f"❌ Step 9 test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())







