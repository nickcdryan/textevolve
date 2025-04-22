#!/usr/bin/env python
"""
validate_script.py - Validate a script on a specific range of examples
"""
import os
import sys
import json
import argparse
from pathlib import Path
from agent_system import AgentSystem

def main():
    parser = argparse.ArgumentParser(description="Validate a script on a specific range of examples")
    parser.add_argument("--script", "-s", type=str, help="Path to script (default: best script)")
    parser.add_argument("--start", "-b", type=int, default=900, help="Start index (default: 900)")
    parser.add_argument("--end", "-e", type=int, default=999, help="End index (default: 999)")
    parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed results")
    parser.add_argument("--dataset", "-f", type=str, default="calendar_scheduling.json", 
                      help="Path to dataset file (default: calendar_scheduling.json)")
    parser.add_argument("--prefix", "-p", type=str, default="calendar_scheduling_example_",
                      help="Prefix for example keys (default: calendar_scheduling_example_)")
    args = parser.parse_args()

    # Check environment variables
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable is not set.")
        print("Please set this variable to your Gemini API key before running the script.")
        print("Example: export GEMINI_API_KEY=your_api_key_here")
        sys.exit(1)

    # Initialize the agent system
    try:
        agent = AgentSystem(dataset_path=args.dataset, example_prefix=args.prefix)
    except Exception as e:
        print(f"Error initializing agent system: {e}")
        sys.exit(1)

    # In validate_script.py, update the section that prints best script info
    # If no script specified, print best script info
    if not args.script:
        best_script_info = agent.get_best_script_info()
        if best_script_info:
            print("\n=== Current Best Script ===")
            print(f"Iteration: {best_script_info.get('iteration')}")

            # Report batch testing results
            batch_acc = best_script_info.get('accuracy', 0)
            batch_size = best_script_info.get('batch_size', 0)
            print(f"Batch Accuracy: {batch_acc:.2f} (tested on {batch_size} examples)")

            # Report progressive testing results if available
            prog_acc = best_script_info.get('progressive_accuracy')
            if prog_acc is not None:
                prog_samples = best_script_info.get('progressive_samples', 0)
                print(f"Progressive Accuracy: {prog_acc:.2f} (tested on {prog_samples} examples)")

            # Report combined accuracy if available
            combined_acc = best_script_info.get('combined_accuracy')
            if combined_acc is not None:
                total_samples = batch_size
                if prog_acc is not None:
                    total_samples += best_script_info.get('progressive_samples', 0)
                print(f"Combined Accuracy: {combined_acc:.2f} (across all {total_samples} examples)")

            print(f"Path: {best_script_info.get('path')}")
            print(f"Approach: {best_script_info.get('approach')}")
            print(f"Rationale: {best_script_info.get('rationale')}")
            script_path = best_script_info.get('path')
        else:
            print("No scripts available.")
            sys.exit(1)
    else:
        script_path = args.script

    print(f"\nValidating script: {script_path}")
    print(f"Example range: {args.start} to {args.end}")

    # Run validation
    result = agent.validate_script(script_path, args.start, args.end)

    # Print results
    if result.get("success") == False:
        print(f"Error: {result.get('error')}")
        sys.exit(1)

    print("\n=== Validation Results ===")
    print(f"Total examples: {result.get('total_examples', 0)}")
    print(f"Successful runs: {result.get('successful_runs', 0)}")
    print(f"Correct answers: {result.get('matches', 0)}")
    print(f"Accuracy: {result.get('accuracy', 0):.2f}")

    # Show detailed results if requested
    if args.detailed and result.get("results"):
        print("\n=== Detailed Results ===")
        for i, item in enumerate(result.get("results", [])):
            result_data = item.get("result", {})
            success = result_data.get("success", False)
            match = result_data.get("match", False)
            status = "✅" if match else "❌"
            if not success:
                status = "⚠️"
            print(f"{status} {item.get('key')}: {'Success' if success else 'Error'}, {'Match' if match else 'No match'}")
            if not success:
                print(f"   Error: {result_data.get('error', 'Unknown error')}")
            elif not match and "evaluation" in result_data:
                print(f"   Explanation: {result_data.get('evaluation', {}).get('explanation', 'No explanation')}")

if __name__ == "__main__":
    main()