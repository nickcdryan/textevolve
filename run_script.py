#!/usr/bin/env python
"""
run.py - Main entry point for the Agentic Learning System
"""

import os
import sys
import json
import random
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Any

from agent_system import AgentSystem
from capability_debug import initialize_debug_agent
from debug_capability import patch_agent_system, DebuggingCapabilityTracker

# Fixed random seed for reproducible dataset shuffling
RANDOM_SEED = 42


def verify_dataset(dataset_path: str, example_prefix: str) -> bool:
    """
    Verify that the dataset exists and has the expected format.
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file {dataset_path} does not exist.")
        return False

    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check for expected fields in at least one example
        example_key = f"{example_prefix}0"
        if example_key not in data:
            print(
                f"Error: Expected to find key '{example_key}' in dataset, but it's missing."
            )
            return False

        sample = data[example_key]
        if "prompt_0shot" not in sample or "golden_plan" not in sample:
            print(
                "Error: Dataset examples should contain 'prompt_0shot' and 'golden_plan' fields."
            )
            return False

        # Count examples
        example_count = sum(1 for key in data
                            if key.startswith(example_prefix))
        print(
            f"Dataset verification successful. Found {example_count} examples with required fields."
        )
        return True
    except Exception as e:
        print(f"Error verifying dataset: {e}")
        return False


def shuffle_dataset(dataset_path: str, example_prefix: str) -> str:
    """
    Shuffle the dataset examples using a fixed random seed.
    Returns the path to the shuffled dataset file.
    """
    # Set fixed random seed for reproducibility
    random.seed(RANDOM_SEED)

    try:
        # Load the original dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Identify all example keys
        example_keys = [
            key for key in data.keys() if key.startswith(example_prefix)
        ]

        # Extract all examples
        examples = [data[key] for key in example_keys]

        # Shuffle the examples
        random.shuffle(examples)

        # Create a new dataset with shuffled examples
        shuffled_data = {}
        for i, example in enumerate(examples):
            shuffled_data[f"{example_prefix}{i}"] = example

        # Save the shuffled dataset
        shuffled_path = f"shuffled_{dataset_path}"
        with open(shuffled_path, 'w', encoding='utf-8') as f:
            json.dump(shuffled_data, f, indent=2)

        print(f"Dataset shuffled successfully. Saved to {shuffled_path}")
        return shuffled_path

    except Exception as e:
        print(f"Error shuffling dataset: {e}")
        print("Using original dataset instead.")
        return dataset_path


def run_agent(iterations: int,
              dataset_path: str = "calendar_scheduling.json",
              example_prefix: str = "calendar_scheduling_example_") -> None:
    """
    Run the agent system for the specified number of iterations.
    """
    # Verify the dataset format
    if not verify_dataset(dataset_path, example_prefix):
        print(
            "Dataset verification failed. Please check the format and try again."
        )
        sys.exit(1)

    # Shuffle the dataset with fixed random seed
    print(f"Shuffling dataset with random seed {RANDOM_SEED}...")
    shuffled_dataset_path = shuffle_dataset(dataset_path, example_prefix)

    # Initialize the agent system with the shuffled dataset
    try:
        # Apply enhanced capability debugging
        print("Applying enhanced capability debugging...")
        patch_agent_system()
        
        # Use the debug-enabled agent
        print("Initializing debug-enabled agent system...")
        agent = initialize_debug_agent(dataset_path=shuffled_dataset_path,
                                      example_prefix=example_prefix)
        
        # Verify the capability tracker type
        tracker_type = type(agent.capability_tracker).__name__
        print(f"Capability tracker type: {tracker_type}")
        if not isinstance(agent.capability_tracker, DebuggingCapabilityTracker):
            print("Warning: DebuggingCapabilityTracker not properly installed.")
            print("Manually applying debugging capability tracker...")
            agent.capability_tracker = DebuggingCapabilityTracker()
    except Exception as e:
        print(f"Error initializing agent system: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("=" * 80)
    print("Agentic Learning System")
    print("=" * 80)
    print(
        f"Starting with explore/exploit balance: {agent.explore_rate}/{agent.exploit_rate}"
    )
    print(f"Starting batch size: {agent.current_batch_size}")
    print("-" * 80)

    # Run iterations
    for i in range(iterations):
        try:
            result = agent.run_iteration()
            if not result.get("success", True):
                print(
                    f"Iteration {i} failed: {result.get('error', 'Unknown error')}"
                )
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

        # Print performance trend
        print("\nPerformance Trend:")
        print(
            f"{'Iteration':<10} {'Strategy':<12} {'Accuracy':<10} {'Batch':<5} {'Explore/Exploit':<15} {'Primary Issue'}"
        )
        print("-" * 80)

        for summary in summaries:
            iteration = summary.get("iteration", "?")
            strategy = summary.get("strategy", "Unknown")
            accuracy = summary.get("performance", {}).get("accuracy", 0) * 100
            batch_size = summary.get("batch_size", 5)
            explore = summary.get("explore_rate", 0)
            exploit = summary.get("exploit_rate", 0)
            prog_accuracy = summary.get("progressive_accuracy", None)
            issue = summary.get("primary_issue", "None identified")

            # Truncate issue if too long
            if len(issue) > 30:
                issue = issue[:27] + "..."

            # Add indicator for progressive testing results
            accuracy_str = f"{accuracy:<10.2f}%"
            if prog_accuracy is not None:
                accuracy_str = f"{accuracy:<4.2f}% ({prog_accuracy:.2f}%)"

            print(
                f"{iteration:<10} {strategy:<12} {accuracy_str:<15} {batch_size:<5} {explore}/{exploit:<15} {issue}"
            )

    # Get best script info - with error handling
    try:
        best_script_info = agent.get_best_script_info()
        if best_script_info:
            print("\n=== Current Best Script ===")
            print(f"Iteration: {best_script_info.get('iteration')}")
            print(
                f"Accuracy: {best_script_info.get('accuracy', 0):.2f} (tested on {best_script_info.get('batch_size', 0)} examples)"
            )
            print(f"Path: {best_script_info.get('path')}")
            print(f"Approach: {best_script_info.get('approach')}")
            print(f"Rationale: {best_script_info.get('rationale')}")
            print(
                "\nTo validate this script on a specific range of examples, run:"
            )
            print(
                f"python validate_script.py --script {best_script_info.get('path')} --start 900 --end 999"
            )
    except Exception as e:
        print(f"Error getting best script info: {e}")
        print("Could not determine best script due to an error.")

    # Final explore/exploit balance and batch size
    print(
        f"\nFinal explore/exploit balance: {agent.explore_rate}/{agent.exploit_rate}"
    )
    print(f"Final batch size: {agent.current_batch_size}")
    print(f"Total examples seen: {len(agent.seen_examples)}")
    print("=" * 80)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Agentic Learning System")
    parser.add_argument("--iterations",
                        "-i",
                        type=int,
                        default=5,
                        help="Number of iterations to run (default: 5)")
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="calendar_scheduling.json",
        help="Path to the dataset file (default: calendar_scheduling.json)")
    parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        default="calendar_scheduling_example_",
        help=
        "Prefix for example keys in the dataset (default: calendar_scheduling_example_)"
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for dataset shuffling (default: {RANDOM_SEED})")

    return parser.parse_args()

def set_random_seed(new_seed):
    """Set the global random seed."""
    global RANDOM_SEED
    RANDOM_SEED = new_seed
    print(f"Using random seed: {RANDOM_SEED}")

if __name__ == "__main__":

    # Parse command-line arguments
    args = parse_arguments()

    # Update the random seed if specified
    if args.seed != RANDOM_SEED:

        set_random_seed(args.seed)
        print(f"Using custom random seed: {RANDOM_SEED}")

    # Check environment variables
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable is not set.")
        print(
            "Please set this variable to your Gemini API key before running the script."
        )
        print("Example: export GEMINI_API_KEY=your_api_key_here")
        sys.exit(1)

    # Run the agent
    run_agent(args.iterations, args.dataset, args.prefix)
