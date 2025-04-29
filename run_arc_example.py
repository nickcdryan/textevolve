#!/usr/bin/env python
"""
run_arc_example.py - Example script for running the Agentic Learning System on the ARC dataset
"""

import os
import sys
from dataset_loader import ARCDatasetLoader
from agent_system import AgentSystem

def main():
    """Run the Agentic Learning System on the ARC dataset"""

    # Check for API key
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable is not set.")
        print("Please set it with: export GEMINI_API_KEY=your_api_key_here")
        sys.exit(1)

    # Check if the ARC dataset directory exists
    arc_dir = "ARC_2024_Training"
    if not os.path.isdir(arc_dir):
        print(f"Error: {arc_dir} directory not found.")
        print("Please make sure the ARC dataset is available.")
        sys.exit(1)

    # Create the ARC dataset loader
    print(f"Creating ARC dataset loader from {arc_dir}...")
    try:
        loader = ARCDatasetLoader(
            dataset_path=arc_dir,
            shuffle=True,
            random_seed=42
        )
        print(f"Successfully loaded {loader.get_total_count()} examples from ARC dataset")

        # Print a few sample examples
        print("\nSample examples from ARC dataset:")
        samples = loader.get_examples(3)
        for i, sample in enumerate(samples):
            print(f"Example {i+1}:")
            print(f"  Input: {loader.get_example_input(sample)}")
            print(f"  Output: {loader.get_example_output(sample)}")

        # Initialize the agent system
        print("\nInitializing Agentic Learning System...")
        agent = AgentSystem(dataset_loader=loader)

        # Run a few iterations
        num_iterations = 3
        print(f"\nRunning {num_iterations} iterations...")

        for i in range(num_iterations):
            print(f"\n=== Starting Iteration {i+1}/{num_iterations} ===")
            try:
                result = agent.run_iteration()
                if not result.get("success", True):
                    print(f"Iteration {i+1} failed: {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"Error in iteration {i+1}: {e}")
                import traceback
                traceback.print_exc()

        # Print summary
        print("\n=== Summary ===")
        print(f"Completed {num_iterations} iterations on the ARC dataset")
        print(f"Best script is in the scripts directory")
        print("Use 'python run_script.py' for a full run with more iterations")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()