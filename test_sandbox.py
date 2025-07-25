#!/usr/bin/env python
"""
Test script for Docker sandbox functionality.
"""

import sys
from sandbox import DockerSandbox, check_docker_available

def test_docker_availability():
    """Test if Docker is available."""
    print("Testing Docker availability...")
    available = check_docker_available()
    print(f"Docker available: {available}")
    return available

def test_sandbox_creation():
    """Test creating a Docker sandbox instance."""
    print("\nTesting sandbox creation...")
    try:
        sandbox = DockerSandbox()
        print("✓ Sandbox created successfully")
        return sandbox
    except Exception as e:
        print(f"✗ Failed to create sandbox: {e}")
        return None

def test_script_execution(sandbox):
    """Test executing a simple script in the sandbox."""
    print("\nTesting script execution...")
    
    test_script = '''
import sys

def main():
    print("ANSWER_START")
    print("Hello from sandbox!")
    print("ANSWER_END")

if __name__ == "__main__":
    main()
'''
    
    test_sample = {"question": "Test question", "id": "test_1"}
    
    try:
        result = sandbox.execute_script(test_script, test_sample)
        print(f"✓ Script executed. Success: {result.get('success')}")
        if result.get('success'):
            print(f"  Answer: {result.get('answer')}")
        else:
            print(f"  Error: {result.get('error')}")
        return result
    except Exception as e:
        print(f"✗ Script execution failed: {e}")
        return None

def main():
    """Run all tests."""
    print("=" * 50)
    print("Docker Sandbox Test Suite")
    print("=" * 50)
    
    # Test 1: Docker availability
    if not test_docker_availability():
        print("\nDocker is not available. Cannot proceed with further tests.")
        print("To test sandbox functionality:")
        print("1. Install Docker Desktop")
        print("2. Start Docker")
        print("3. Run this test again")
        return False
    
    # Test 2: Sandbox creation
    sandbox = test_sandbox_creation()
    if not sandbox:
        return False
    
    # Test 3: Image availability
    print("\nTesting image availability...")
    if not sandbox.ensure_image_available():
        print("✗ Failed to ensure Docker image is available")
        return False
    print("✓ Docker image available")
    
    # Test 4: Script execution
    result = test_script_execution(sandbox)
    if not result:
        return False
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("Docker sandbox is ready for use.")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)