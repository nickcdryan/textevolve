#!/usr/bin/env python
"""
run_script.py - Main entry point for the Agentic Learning System with custom dataset loaders

python run_script.py --iterations 5 --dataset ARC_2024_Training/ --loader arc --no-shuffle

python run_script.py --iterations 5 --dataset ARC_2024_Training/ --loader arc
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

from agent_system import AgentSystem
from dataset_loader import create_dataset_loader

# Fixed random seed for reproducibility (if shuffling is enabled)
RANDOM_SEED = 42

def run_agent(iterations: int, loader_config: Dict) -> None:
    """
    Run the agent system for the specified number of iterations.

    Args:
        iterations: Number of iterations to run
        loader_config: Configuration for dataset loader
    """
    # Create the appropriate dataset loader
    try:
        loader_type = loader_config.pop("loader_type")
        dataset_loader = create_dataset_loader(loader_type, **loader_config)
        print(f"Created {loader_type} dataset loader with {dataset_loader.get_total_count()} examples")

        # Initialize the agent system with the dataset loader
        agent = AgentSystem(dataset_loader=dataset_loader)
    except Exception as e:
        print(f"Error initializing system: {e}")
        sys.exit(1)

    print("=" * 80)
    print("Agentic Learning System")
    print("=" * 80)
    print(f"Dataset: {loader_config.get('dataset_path')}")
    print(f"Loader type: {loader_type}")
    print(f"Shuffle data: {loader_config.get('shuffle', True)}")
    print(f"Starting with explore/exploit/refine balance: {agent.explore_rate}/{agent.exploit_rate}/{agent.refine_rate}")
    print(f"Starting batch size: {agent.current_batch_size}")
    print("-" * 80)

    # Run iterations
    for i in range(iterations):
        try:
            result = agent.run_iteration()
            if not result.get("success", True):
                print(f"Iteration {i} failed: {result.get('error', 'Unknown error')}")
                break
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Saving current state...")
            break
        except Exception as e:
            print(f"\nError in iteration {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print final summary
    print("\n" + "=" * 80)
    print("Final Results Summary")
    print("=" * 80)

    summaries = agent.get_summaries()

    if summaries:
        # Sort by iteration number
        summaries.sort(key=lambda x: x.get("iteration", 0))

        all_iteration_data = agent.get_all_iterations()

        # Print performance trend
        print("\nPerformance Trend:")
        print(f"{'Iteration':<8} {'Strategy':<12} {'Batch Acc.':<12} {'Prog. Acc.':<16} {'Combined':<12} {'Batch Size':<10} {'Prog. Size':<10} {'Expl/Expt':<10} {'Primary Issue'}")
        print("-" * 120)

        for summary in summaries:
            iteration = summary.get("iteration", "?")
            strategy = summary.get("strategy", "Unknown")
            batch_accuracy = summary.get("performance", {}).get("accuracy", 0) * 100
            batch_size = summary.get("batch_size", 5)
            explore = summary.get("explore_rate", 0)
            exploit = summary.get("exploit_rate", 0)
            refine = summary.get("refine_rate", 0)
            prog_accuracy = summary.get("progressive_accuracy", None)

            # Get progressive testing sample count
            prog_samples = 0
            for it in all_iteration_data:
                if it and it.get("iteration") == summary.get("iteration"):
                    if "progressive_testing" in it and it["progressive_testing"]:
                        prog_samples = it["progressive_testing"].get("total_examples", 0)
                        break

            issue = summary.get("primary_issue", "None identified")

            # Truncate issue if too long
            if len(issue) > 30:
                issue = issue[:27] + "..."

            # Format progressive accuracy with sample count
            if prog_accuracy is not None:
                prog_acc_str = f"{prog_accuracy*100:.2f}% ({prog_samples})"
            else:
                prog_acc_str = "N/A"

            # Calculate combined accuracy - weighted by sample counts
            combined_acc_str = "N/A"
            if prog_accuracy is not None and batch_accuracy > 0:
                # Correct weighted average calculation
                total_correct = (batch_accuracy/100 * batch_size) + (prog_accuracy * prog_samples)
                total_samples = batch_size + prog_samples
                combined_acc = (total_correct / total_samples) * 100
                combined_acc_str = f"{combined_acc:.2f}%"

            print(f"{iteration:<8} {strategy:<12} {batch_accuracy:<12.2f}% {prog_acc_str:<16} {combined_acc_str:<12} {batch_size:<10} {prog_samples:<10} {explore}/{exploit}/{refine:<10} {issue}")

    # Get best script info - with error handling
    try:
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
            print("\nTo validate this script on a specific range of examples, run:")
            print(f"python validate_script.py --script {best_script_info.get('path')}")
    except Exception as e:
        print(f"Error getting best script info: {e}")
        print("Could not determine best script due to an error.")


    print(f"Final batch size: {agent.current_batch_size}")
    print(f"Total examples seen: {len(agent.seen_examples)}")
    print("=" * 80)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Agentic Learning System with custom dataset loaders")

    parser.add_argument("--iterations",
                        "-i",
                        type=int,
                        default=5,
                        help="Number of iterations to run (default: 5)")

    # Dataset configuration
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="dataset.json",
        help="Path to the dataset file or directory (default: dataset.json)")


    parser.add_argument(
            "--loader",
            "-l",
            type=str,
            choices=["arc", "json", "jsonl", "simpleqa", "custom", "natural_plan", "hotpotqa", "math", "gpqa"],  
            default="arc",
            help="Type of dataset loader to use (default: arc)")

    # JSON loader options
    parser.add_argument(
        "--input-field",
        "-if",
        type=str,
        default="input",
        help="Field name for input data in JSON/JSONL loader (default: input)")

    parser.add_argument(
        "--output-field",
        "-of",
        type=str,
        default="output",
        help="Field name for output data in JSON/JSONL loader (default: output)")

    parser.add_argument(
        "--example-prefix",
        "-p",
        type=str,
        default="",
        help="Prefix for example keys in JSON loader (default: none)")

    # JSONL loader options
    parser.add_argument(
        "--passage-field",
        type=str,
        default="passage",
        help="Field name for passage text in JSONL loader (default: passage)")

    parser.add_argument(
        "--answer-extraction",
        type=str,
        default="spans",
        help="Field to extract from nested answer data in JSONL loader (default: spans)")

    # General options
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable dataset shuffling (default: False)")

    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for dataset shuffling (default: {RANDOM_SEED})")

    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Check environment variables
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable is not set.")
        print(
            "Please set this variable to your Gemini API key before running the script."
        )
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

    # Run the agent
    run_agent(args.iterations, loader_config)