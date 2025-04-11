#!/usr/bin/env python
"""
debug_script.py - Debug and fix script execution issues
"""

import os
import sys
import inspect
import importlib.util
import argparse
from pathlib import Path

def debug_script(script_path):
    """Debug a script by checking for common issues and fixing them if possible"""
    if not os.path.exists(script_path):
        print(f"Error: Script file {script_path} does not exist.")
        return False

    print(f"Analyzing script: {script_path}")

    # Read the script content
    with open(script_path, 'r', encoding='utf-8') as f:
        script_content = f.read()

    # Check if the script has a 'main' function
    has_main_function = "def main(" in script_content
    if not has_main_function:
        print("Issue detected: Script does not have a 'main' function")

        # Check if there's a similar function that could be renamed
        possible_main_functions = []
        for line in script_content.split('\n'):
            if line.strip().startswith("def ") and "(" in line:
                function_name = line.strip().split("def ")[1].split("(")[0].strip()
                if function_name != "main" and ("solve" in function_name or 
                                               "process" in function_name or 
                                               "schedule" in function_name or
                                               function_name.lower() == "process_question"):
                    possible_main_functions.append(function_name)

        if possible_main_functions:
            print(f"Found potential main function: {possible_main_functions[0]}")

            # Modify the script to add a main function that calls the found function
            new_content = script_content + f"\n\ndef main(question):\n    return {possible_main_functions[0]}(question)\n"

            # Save to a backup file
            backup_path = script_path + ".backup"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(script_content)
            print(f"Original script backed up to: {backup_path}")

            # Save the modified script
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Added 'main' function wrapper for {possible_main_functions[0]}")
            return True
        else:
            print("No suitable function found to rename as 'main'")
            return False
    else:
        print("Script has a 'main' function - no fix needed")
        return True

def test_script(script_path, test_input=None):
    """Test if a script is importable and has a working main function"""
    script_dir = os.path.dirname(os.path.abspath(script_path))
    script_name = os.path.basename(script_path)
    module_name = os.path.splitext(script_name)[0]

    # Add script directory to path
    sys.path.insert(0, script_dir)

    try:
        # Try to import the script as a module
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Check if it has a main function
        if hasattr(module, 'main'):
            print(f"Successfully imported {module_name} and found 'main' function")

            # If test input is provided, test the main function
            if test_input:
                print(f"Testing main function with input: {test_input}")
                try:
                    result = module.main(test_input)
                    print(f"Result: {result}")
                except Exception as e:
                    print(f"Error executing main function: {e}")
                    return False

            return True
        else:
            print(f"Module {module_name} does not have a 'main' function")
            return False
    except Exception as e:
        print(f"Error importing module: {e}")
        return False
    finally:
        # Remove script directory from path
        if script_dir in sys.path:
            sys.path.remove(script_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug and fix script execution issues")
    parser.add_argument("script", help="Path to the script file to debug")
    parser.add_argument("--test", "-t", action="store_true", help="Test the script after debugging")
    parser.add_argument("--input", "-i", type=str, help="Test input for the main function")

    args = parser.parse_args()

    # Debug the script
    success = debug_script(args.script)

    # Test the script if requested
    if args.test and success:
        test_script(args.script, args.input)