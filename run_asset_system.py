#!/usr/bin/env python
"""
run_asset_system.py - Runner for asset-based datasets using EnhancedAgentSystem

Usage:
    python run_asset_system.py --dataset synthetic_data/customer_service_dataset --iterations 10
    python run_asset_system.py --dataset synthetic_data/customer_service_dataset --iterations 5 --use-sandbox
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

from enhanced_agent_system import EnhancedAgentSystem
from asset_dataset_loader import AssetDatasetLoader

# Fixed random seed for reproducibility
RANDOM_SEED = 42

def load_customer_service_examples(dataset_path: str) -> List[Dict]:
    """Load customer service evaluation examples"""
    # dataset_path is the full path to evaluation_data.json
    eval_file = Path(dataset_path)
    
    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation data not found: {eval_file}")
    
    with open(eval_file, 'r') as f:
        data = json.load(f)
    
    # Convert the flat dictionary structure to a list of examples
    examples = []
    for ticket_id, ticket_data in data.items():
        if isinstance(ticket_data, dict) and "question" in ticket_data and "answer" in ticket_data:
            example = {
                "id": ticket_id,
                "ticket": ticket_data.get("question", ""),
                "resolution_plan": ticket_data.get("answer", "")
            }
            examples.append(example)
    
    return examples

def get_customer_service_input(example: Dict) -> str:
    """Extract input (customer service ticket) from example - ONLY the ticket, no context data"""
    
    # Get the full question which contains ticket + database + policy
    full_question = example.get("ticket", "")
    
    # Extract just the ticket part - everything before "CUSTOMER DATABASE:"
    if "CUSTOMER DATABASE:" in full_question:
        ticket_only = full_question.split("CUSTOMER DATABASE:")[0].strip()
    else:
        ticket_only = full_question
    
    # Remove the "TASK:" section if present
    if "TASK:" in ticket_only:
        ticket_only = ticket_only.split("TASK:")[0].strip()
    
    return ticket_only

def get_customer_service_output(example: Dict) -> str:
    """Extract expected output (resolution plan) from example"""
    return example.get("resolution_plan", "")

def run_asset_system(dataset_path: str, iterations: int, use_sandbox: bool = False) -> None:
    """
    Run the enhanced agent system on asset-based dataset
    
    Args:
        dataset_path: Path to the asset-based dataset directory
        iterations: Number of iterations to run
        use_sandbox: Whether to use Docker sandbox for code execution
    """
    print("🚀 Starting TextEvolve Asset-Based System")
    print("=" * 60)
    
    try:
        # Create asset-enabled dataset loader
        dataset_loader = AssetDatasetLoader(
            dataset_path=str(Path(dataset_path) / "evaluation_data.json"),
            load_examples_fn=load_customer_service_examples,
            get_input_fn=get_customer_service_input,
            get_output_fn=get_customer_service_output,
            shuffle=True,
            random_seed=RANDOM_SEED,
            config_file="dataset.yaml"
        )
        
        print(f"✅ Loaded asset-based dataset")
        print(f"   📊 Total examples: {dataset_loader.get_total_count()}")
        print(f"   🔧 Available tools: {list(dataset_loader.available_tools.keys())}")
        
        # Create enhanced agent system
        agent = EnhancedAgentSystem(
            dataset_loader=dataset_loader,
            use_sandbox=use_sandbox
        )
        
        print(f"✅ Enhanced Agent System initialized")
        print(f"   🏗️  Asset support: {'Yes' if agent.has_assets else 'No'}")
        print(f"   🐳 Sandbox enabled: {agent.use_sandbox}")
        
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"🎯 Starting {iterations} iterations of improvement")
    print("=" * 60)
    
    try:
        # Run iterations
        for i in range(iterations):
            try:
                print(f"\n🔄 Running iteration {i+1}/{iterations}")
                result = agent.run_iteration()
                if not result.get("success", True):
                    print(f"❌ Iteration {i+1} failed: {result.get('error', 'Unknown error')}")
                    break
                else:
                    print(f"✅ Iteration {i+1} completed successfully")
            except KeyboardInterrupt:
                print(f"\n⏸️  Training interrupted by user at iteration {i+1}")
                break
            except Exception as e:
                print(f"\n❌ Error in iteration {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print("\n" + "=" * 60)
        print("🎉 Asset-based training completed!")
        print("=" * 60)
        
        # Print summary statistics
        summaries = agent.get_summaries()
        if summaries:
            print(f"📈 Performance summary:")
            summaries.sort(key=lambda x: x.get("iteration", 0))
            for summary in summaries[-5:]:  # Show last 5 iterations
                iteration = summary.get("iteration", "?")
                combined_score = summary.get("combined_score", 0)
                strategy = summary.get("strategy", "unknown")
                print(f"   Iteration {iteration}: {combined_score:.3f} ({strategy})")
        
    except KeyboardInterrupt:
        print("\n⏸️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 Cleaning up...")
        agent.cleanup()
        print("✅ Cleanup completed")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run TextEvolve system on asset-based datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_asset_system.py --dataset synthetic_data/customer_service_dataset --iterations 10
  python run_asset_system.py --dataset synthetic_data/customer_service_dataset --iterations 5 --use-sandbox
        """
    )
    
    parser.add_argument(
        "--dataset", 
        required=True,
        help="Path to the asset-based dataset directory"
    )
    
    parser.add_argument(
        "--iterations", 
        type=int, 
        default=10,
        help="Number of improvement iterations to run (default: 10)"
    )
    
    parser.add_argument(
        "--use-sandbox", 
        action="store_true",
        help="Use Docker sandbox for code execution (requires Docker)"
    )
    
    args = parser.parse_args()
    
    # Validate dataset path
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    if not (dataset_path / "dataset.yaml").exists():
        print(f"❌ No dataset.yaml found in: {dataset_path}")
        print("This script requires an asset-based dataset with configuration file")
        sys.exit(1)
    
    # Run the system
    run_asset_system(
        dataset_path=str(dataset_path),
        iterations=args.iterations,
        use_sandbox=args.use_sandbox
    )

if __name__ == "__main__":
    main() 