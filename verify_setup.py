#!/usr/bin/env python3
"""
Setup verification script for the sandbox environment.

This script tests Docker functionality and provides clear feedback about
whether the system is ready for secure code execution.
"""

import sys
import os
import platform
import subprocess
from pathlib import Path
from sandbox import auto_setup_docker, check_docker_available, DockerSandbox


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_status(check: str, status: bool, message: str = ""):
    """Print a status line with checkmark or X."""
    symbol = "✅" if status else "❌"
    print(f"{symbol} {check}")
    if message:
        print(f"   {message}")


def test_docker_basic():
    """Test basic Docker functionality."""
    print_section("DOCKER BASIC TESTS")
    
    # Test 1: Check if Docker is available
    docker_available = check_docker_available()
    print_status("Docker daemon is running", docker_available)
    
    if not docker_available:
        print("\n⚠️  Docker is not available. Attempting auto-setup...")
        success, message = auto_setup_docker()
        print_status("Auto-setup completed", success, message)
        
        if success:
            docker_available = check_docker_available()
            print_status("Docker now available", docker_available)
    
    return docker_available


def test_docker_functionality():
    """Test Docker container functionality."""
    print_section("DOCKER FUNCTIONALITY TESTS")
    
    try:
        # Test 2: Initialize DockerSandbox
        sandbox = DockerSandbox(auto_setup=False)  # Don't auto-setup again
        print_status("DockerSandbox initialization", True)
        
        # Test 3: Check if image is available
        image_available = sandbox.ensure_image_available()
        print_status(f"Docker image ({sandbox.image}) is available", image_available)
        
        if not image_available:
            print("   This may take a few minutes on first run...")
            return False
        
        # Test 4: Run a simple test script
        test_script = '''
import os
print("ANSWER_START")
print("Hello from Docker sandbox!")
print("ANSWER_END")
'''
        
        test_sample = {"question": "test", "answer": "test", "id": "test"}
        result = sandbox.execute_script(test_script, test_sample)
        
        success = result.get("success", False)
        print_status("Sample script execution", success)
        
        if success:
            answer = result.get("answer", "")
            print_status("Expected output format", "Hello from Docker sandbox!" in answer)
        else:
            print(f"   Error: {result.get('error', 'Unknown error')}")
        
        return success
        
    except Exception as e:
        print_status("DockerSandbox functionality", False, f"Error: {e}")
        return False


def test_fallback_mode():
    """Test fallback to direct execution."""
    print_section("FALLBACK MODE TEST")
    
    # This would test the non-Docker execution path
    # For now, just check if the system can run Python scripts
    try:
        result = subprocess.run([sys.executable, "-c", "print('Python execution works')"], 
                              capture_output=True, text=True, timeout=10)
        success = result.returncode == 0
        print_status("Direct Python execution", success)
        return success
    except Exception as e:
        print_status("Direct Python execution", False, f"Error: {e}")
        return False


def get_system_info():
    """Get and display system information."""
    print_section("SYSTEM INFORMATION")
    
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python Version: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Check for common virtualization indicators
    virtualizations = []
    if os.path.exists("/.dockerenv"):
        virtualizations.append("Docker container")
    if os.environ.get("AWS_EXECUTION_ENV"):
        virtualizations.append("AWS Lambda")
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        virtualizations.append("Kubernetes")
    
    if virtualizations:
        print(f"Detected Environment: {', '.join(virtualizations)}")
    else:
        print("Detected Environment: Native system")


def main():
    """Main verification function."""
    print("🔧 TextEvolve Sandbox Setup Verification")
    print("This script will test the sandbox environment setup.")
    
    get_system_info()
    
    # Test Docker functionality
    docker_works = test_docker_basic()
    
    if docker_works:
        docker_functional = test_docker_functionality()
    else:
        docker_functional = False
    
    # Test fallback mode
    fallback_works = test_fallback_mode()
    
    # Summary
    print_section("SUMMARY")
    
    if docker_functional:
        print("🎉 SUCCESS: Docker sandbox is fully functional!")
        print("   Your system is ready for secure code execution.")
        return 0
    elif docker_works:
        print("⚠️  PARTIAL: Docker is available but not fully functional.")
        print("   Some advanced features may not work correctly.")
        if fallback_works:
            print("   Fallback mode is available as backup.")
        return 1
    elif fallback_works:
        print("⚠️  FALLBACK: Docker not available, using direct execution.")
        print("   ⚠️  WARNING: This provides NO SECURITY ISOLATION!")
        print("   Only use this mode in trusted environments.")
        return 2
    else:
        print("❌ FAILURE: Neither Docker nor fallback mode is working.")
        print("   System is not ready for code execution.")
        return 3


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)