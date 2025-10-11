#!/usr/bin/env python
"""
validate_script_fix.py - A version of validate_script.py with a fix for the string vs dictionary issue
"""
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from agent_system import AgentSystem
from dataset_loader import create_dataset_loader

def create_validation_results_dir():
    """Create validation_results directory if it doesn't exist."""
    results_dir = Path("./validation_results")
    results_dir.mkdir(exist_ok=True)
    return results_dir

def generate_results_filename(script_path, dataset_path, loader_type, start_idx, end_idx):
    """Generate a unique filename for the validation results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = Path(script_path).stem
    dataset_name = Path(dataset_path).stem
    filename = f"{timestamp}_{script_name}_{dataset_name}_{loader_type}_{start_idx}-{end_idx}.json"
    return filename

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
                        choices=["arc", "json", "jsonl", "custom", "simpleqa", "natural_plan", "hotpotqa", "math", "gpqa", "medmcqa", "ticketworld", "ticketworld_simple"],
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
    
    # Evaluator options
    parser.add_argument("--evaluator", "-ev", type=str, 
                        choices=["llm", "f1", "exact_match", "exact", "ticketworld"],
                        help="Override the default evaluator for this dataset type")

    # LLM configuration options
    parser.add_argument(
        "--orchestrator-llm",
        type=str,
        default=None,
        help="Orchestrator LLM model name (e.g., gemini-2.5-flash, gemini-1.5-pro). "
             "Used for evaluation and feedback. "
             "Overrides ORCHESTRATOR_LLM_MODEL environment variable.")
    
    parser.add_argument(
        "--inference-llm",
        type=str,
        default=None,
        help="Inference LLM model name (e.g., gemini-2.0-flash, gemini-1.5-flash). "
             "Used for script execution. "
             "Overrides INFERENCE_LLM_MODEL environment variable.")

    args = parser.parse_args()

    # Check environment variables
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable is not set.")
        print("Please set this variable to your Gemini API key before running the script.")
        print("Example: export GEMINI_API_KEY=your_api_key_here")
        sys.exit(1)

    # Set LLM models if specified
    if args.orchestrator_llm:
        os.environ["ORCHESTRATOR_LLM_MODEL"] = args.orchestrator_llm
        print(f"Orchestrator LLM model set to: {args.orchestrator_llm}")
    
    if args.inference_llm:
        os.environ["INFERENCE_LLM_MODEL"] = args.inference_llm
        print(f"Inference LLM model set to: {args.inference_llm}")

    # Create validation results directory and prepare filename
    results_dir = create_validation_results_dir()
    results_filename = generate_results_filename(args.script, args.dataset, args.loader, args.start, args.end)
    results_filepath = results_dir / results_filename

    # Create loader configuration
    loader_config = {
        "loader_type": args.loader,
        "dataset_path": args.dataset,
        "shuffle": not args.no_shuffle,
        "random_seed": args.seed
    }
    
    # Add evaluator override if specified
    if args.evaluator:
        loader_config["evaluator"] = args.evaluator

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

        # Initialize agent system with dataset loader (skip training setup for validation)
        agent = AgentSystem(
            dataset_loader=dataset_loader,
            orchestrator_llm_model=args.orchestrator_llm,
            skip_training_setup=True
        )
    except Exception as e:
        print(f"Error initializing agent system: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    script_path = args.script
    print(f"\nValidating script: {script_path}")
    print(f"Example range: {args.start} to {args.end}")
    print(f"Dataset: {args.dataset} (using {args.loader} loader)")
    print(f"Evaluator: {dataset_loader.get_evaluator()}" + (" (overridden)" if args.evaluator else f" (default for {args.loader})"))

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
            result = agent.execute_script_simplified(script_content, sample)

            # Evaluate the result if successful
            if result.get("success"):
                golden_answer = dataset_loader.get_example_output(sample)
                system_answer = result.get("answer", "")

                # Use configured evaluator
                evaluation = agent.evaluator.evaluate(system_answer, golden_answer, context=sample)
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
            
            # Store detailed result information
            detailed_result = {
                "example_id": sample.get("id", f"example_{i}"),
                "example_index": i,
                "input_data": sample,
                "system_output": result.get("answer", ""),
                "golden_answer": result.get("golden_answer", ""),
                "success": result.get("success", False),
                "match": result.get("match", False),
                "error": result.get("error", ""),
                "evaluation": result.get("evaluation", {}),
                "execution_details": result
            }
            results.append(detailed_result)

        # Restore original index
        dataset_loader.current_index = original_index
    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()

    # Calculate accuracy
    accuracy = matches / total_examples if total_examples > 0 else 0

    # Get model information
    orchestrator_model = os.environ.get("ORCHESTRATOR_LLM_MODEL", "default")
    inference_model = os.environ.get("INFERENCE_LLM_MODEL", "default")
    
    # Get evaluator information
    evaluator_type = dataset_loader.get_evaluator()
    evaluator_info = {
        "type": evaluator_type,
        "overridden": args.evaluator is not None,
        "default_for_dataset": dataset_loader.__class__.default_evaluator if hasattr(dataset_loader.__class__, 'default_evaluator') else "llm"
    }

    # Create comprehensive result object with metadata
    validation_result = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "script_path": script_path,
            "script_contents": script_content,
            "dataset_path": args.dataset,
            "loader_type": args.loader,
            "validation_range": {
                "start": args.start,
                "end": args.end
            },
            "models": {
                "orchestrator_llm": orchestrator_model,
                "inference_llm": inference_model
            },
            "evaluator": evaluator_info,
            "arguments": vars(args),
            "loader_config": loader_config
        },
        "summary": {
            "total_examples": total_examples,
            "successful_runs": successful_runs,
            "matches": matches,
            "accuracy": accuracy,
            "success_rate": successful_runs / total_examples if total_examples > 0 else 0
        },
        "detailed_results": results
    }

    # Write results to file
    try:
        with open(results_filepath, 'w') as f:
            json.dump(validation_result, f, indent=2, default=str)
        print(f"\n📁 Results saved to: {results_filepath}")
    except Exception as e:
        print(f"⚠️ Error saving results to file: {e}")

    # Print results
    print("\n=== Validation Results ===")
    print(f"Inference Model: {inference_model}")
    print(f"Orchestrator Model: {orchestrator_model}")
    print(f"Evaluator: {evaluator_type}" + (" (overridden)" if evaluator_info['overridden'] else f" (default for {args.loader})"))
    print(f"Total examples: {validation_result['summary']['total_examples']}")
    print(f"Successful runs: {validation_result['summary']['successful_runs']}")
    print(f"Correct answers: {validation_result['summary']['matches']}")
    print(f"Accuracy: {validation_result['summary']['accuracy']:.2f}")
    print(f"Success rate: {validation_result['summary']['success_rate']:.2f}")

    # Show detailed results if requested
    if args.detailed and validation_result.get("detailed_results"):
        print("\n=== Detailed Results ===")
        for i, item in enumerate(validation_result.get("detailed_results", [])):
            success = item.get("success", False)
            match = item.get("match", False)
            status = "✅" if match else "❌"
            if not success:
                status = "⚠️"
            print(f"{status} {item.get('example_id')}: {'Success' if success else 'Error'}, {'Match' if match else 'No match'}")
            if not success:
                print(f"   Error: {item.get('error', 'Unknown error')}")
            elif not match and item.get("evaluation"):
                print(f"   Explanation: {item.get('evaluation', {}).get('explanation', 'No explanation')}")

if __name__ == "__main__":
    main()