#!/usr/bin/env python
"""
verify_setup.py - Verify that the system is set up correctly
"""

import os
import sys
import json
from pathlib import Path

def check_api_key():
    """Check if the Gemini API key is set"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable is not set.")
        print("   Please set it with: export GEMINI_API_KEY=your_api_key_here")
        return False
    else:
        print("✅ GEMINI_API_KEY is set.")
        return True

def check_dataset():
    """Check if the dataset file exists and has the expected format"""
    dataset_path = Path("hendrycks_math", "math_test.jsonl")
    example_prefix = "problem"

    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file '{dataset_path}' not found.")
        return False

    try:
        data = {}
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    example = json.loads(line.strip())
                    data[f"{example_prefix}{i}"] = example
                except json.JSONDecodeError as e:
                    print(f"❌ Error parsing line {i+1}: {e}")
                    return False

        # Check if the file contains data
        if not data:
            print(f"❌ Dataset file '{dataset_path}' is empty.")
            return False

        # Check if examples have the required fields
        example = next(iter(data.values()))
        if "problem" not in example or "level" not in example or "type" not in example or "solution" not in example:
            print("❌ Dataset examples are missing required fields.")
            print("   Each example should have 'problem', 'level', 'type', and 'solution' fields.")
            return False

        # Count examples
        count = len(data)
        print(f"✅ Dataset found with {count} examples.")

        # Print a sample
        print("\nSample from dataset:")
        print(f"  problem: {example['problem'][:100]}...")
        print(f"  level: {example['level']}")
        print(f"  type: {example['type']}")
        print(f"  solution: {example['solution'][:100]}...")

        return True
    except Exception as e:
        print(f"❌ Error checking dataset: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import google.genai
        print("✅ Required package 'google-generativeai' is installed.")
        return True
    except ImportError:
        print("❌ Required package 'google-generativeai' is not installed.")
        print("   Install it with: uv pip install google-generativeai")
        return False

def check_directories():
    """Check if required directories exist, create if not"""
    directories = ["archive", "scripts"]
    all_ok = True

    for directory in directories:
        path = Path(directory)
        if not path.exists():
            try:
                path.mkdir()
                print(f"✅ Created missing directory: {directory}")
            except Exception as e:
                print(f"❌ Failed to create directory '{directory}': {e}")
                all_ok = False
        else:
            print(f"✅ Directory exists: {directory}")

    return all_ok

def verify_system_files():
    """Check if all required system files exist"""
    required_files = [
        "agent_system.py",
        "run_script.py"
    ]

    all_ok = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ Required file exists: {file}")
        else:
            print(f"❌ Required file missing: {file}")
            all_ok = False

    return all_ok

def main():
    """Run all verification checks"""
    print("=" * 60)
    print("Agentic Learning System - Setup Verification")
    print("=" * 60)

    api_ok = check_api_key()
    deps_ok = check_dependencies()
    dirs_ok = check_directories()
    files_ok = verify_system_files()
    dataset_ok = check_dataset()

    print("\n" + "=" * 60)
    if api_ok and deps_ok and dirs_ok and files_ok and dataset_ok:
        print("✅ All checks passed! The system is ready to run.")
        print("\nTo start the system, run:")
        print("  python run_script.py -i 5")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above before running the system.")
        return 1

if __name__ == "__main__":
    sys.exit(main())