#!/usr/bin/env python
"""
validate_script_fix.py - A version of validate_script.py with a fix for the string vs dictionary issue
"""
import os
import sys
import json
import argparse
from pathlib import Path
from agent_system import AgentSystem
from dataset_loader import create_dataset_loader

def main():
    parser = argparse.ArgumentParser(description="Validate a script on a specific range of examples")

    # Script selection
    parser.add_argument("--script", "-s", type=str, required=True, 
                        help="Path to script to validate")

    # Validation range
    parser.add_argument("--start", "-b", type=int, default=0, 
                        help="Start index (default: 0)")
    parser.add_argument("--end", "-e", type=int, default=99, 
                        help="End index (default: 99)")
    parser.add_argument("--detailed", "-d", action="store_true", 
                        help="Show detailed results")

    # Dataset options
    parser.add_argument("--dataset", "-f", type=str, required=True,
                        help="Path to dataset file or directory")
    parser.add_argument("--loader", "-l", type=str, 
                        choices=["arc", "json", "jsonl", "custom"],
                        default="arc",
                        help="Type of dataset loader to use (default: arc)")

    # JSON/JSONL loader options
    parser.add_argument("--input-field", "-if", type=str, default="input",
                        help="Field name for input data in JSON/JSONL loader (default: input)")
    parser.add_argument("--output-field", "-of", type=str, default="output",
                        help="Field name for output data in JSON/JSONL loader (default: output)")

    # JSON-specific options
    parser.add_argument("--example-prefix", "-p", type=str, default="",
                        help="Prefix for example keys in JSON loader (default: none)")

    # JSONL-specific options
    parser.add_argument("--passage-field", type=str, default="passage",
                        help="Field name for passage text in JSONL loader (default: passage)")
    parser.add_argument("--answer-extraction", type=str, default="spans",
                        help="Field to extract from nested answer data in JSONL loader (default: spans)")

    # General options
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Disable dataset shuffling (default: False)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for dataset shuffling (default: 42)")

    args = parser.parse_args()

    # Check environment variables
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable is not set.")
        print("Please set this variable to your Gemini API key before running the script.")
        print("Example: export GEMINI_API_KEY=your_api_key_here")
        sys.exit(1)

    # Create loader configuration
    loader_config = {
        "loader_type": args.loader,
        "dataset_path": args.dataset,
        "shuffle": not args.no_shuffle,
        "random_seed": args.seed
    }

    # Add loader-specific parameters
    if args.loader == "json":
        loader_config.update({
            "input_field": args.input_field,
            "output_field": args.output_field
        })
        if args.example_prefix:
            loader_config["example_prefix"] = args.example_prefix

    # Add JSONL loader specific parameters
    elif args.loader == "jsonl":
        loader_config.update({
            "input_field": args.input_field,
            "output_field": args.output_field,
            "passage_field": args.passage_field,
            "answer_extraction": args.answer_extraction
        })

    # Initialize the agent system with dataset loader
    try:
        # Create the dataset loader
        print(f"Creating {args.loader} dataset loader for: {args.dataset}")
        dataset_loader = create_dataset_loader(**loader_config)
        print(f"Loaded dataset with {dataset_loader.get_total_count()} examples")

        # Initialize agent system with dataset loader
        agent = AgentSystem(dataset_loader=dataset_loader)
    except Exception as e:
        print(f"Error initializing agent system: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    script_path = args.script
    print(f"\nValidating script: {script_path}")
    print(f"Example range: {args.start} to {args.end}")
    print(f"Dataset: {args.dataset} (using {args.loader} loader)")

    # Load the script content
    try:
        with open(script_path, 'r') as f:
            script_content = f.read()
    except Exception as e:
        print(f"Error loading script: {e}")
        sys.exit(1)

    # CUSTOM VALIDATION IMPLEMENTATION
    # This section replaces the call to agent.validate_script() which has issues
    print(f"Validating script on {args.end - args.start + 1} examples from range {args.start}-{args.end}...")

    # Manual validation
    results = []
    successful_runs = 0
    matches = 0
    total_examples = 0

    # Get examples in the specified range
    try:
        # Temporarily store current index
        original_index = dataset_loader.current_index

        # Set index to start position
        dataset_loader.current_index = args.start

        # Process examples in the specified range
        for i in range(args.start, args.end + 1):
            examples = dataset_loader.get_examples(1)
            if not examples:
                break

            sample = examples[0]
            total_examples += 1

            # Print progress
            print(f"  Processing sample {total_examples}/{args.end - args.start + 1}...")

            # Execute the script with the sample
            result = agent.execute_script(script_content, sample)

            # Evaluate the result if successful
            if result.get("success"):
                golden_answer = dataset_loader.get_example_output(sample)
                system_answer = result.get("answer", "")

                # Use LLM-based evaluation
                evaluation = agent.evaluate_answer_with_llm(system_answer, golden_answer)
                result["evaluation"] = evaluation
                result["match"] = evaluation.get("match", False)
                result["golden_answer"] = golden_answer

                if result["match"]:
                    matches += 1
                    print(f"    ✅ Match (confidence: {evaluation.get('confidence', 0):.2f})")
                else:
                    print(f"    ❌ No match: {evaluation.get('explanation', '')}")
                print (f"    Total accuracy: {matches/total_examples:.2f}")
            else:
                result["match"] = False
                print(f"    ⚠️ Error: {result.get('error', 'Unknown error')}")

            successful_runs += 1 if result.get("success", False) else 0
            results.append({"key": sample.get("id", f"example_{i}"), "result": result})

        # Restore original index
        dataset_loader.current_index = original_index
    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()

    # Calculate accuracy
    accuracy = matches / total_examples if total_examples > 0 else 0

    # Create result object
    result = {
        "success": True,
        "script_path": script_path,
        "total_examples": total_examples,
        "successful_runs": successful_runs,
        "matches": matches,
        "accuracy": accuracy,
        "results": results
    }

    # Print results
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