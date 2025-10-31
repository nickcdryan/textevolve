#!/usr/bin/env python
"""
Test script for HealthBench dataset loader and evaluator
"""

import sys
from dataset_loader import HealthBenchDatasetLoader
from evaluators import HealthBenchEvaluator
from llm_client import LLMClientFactory

def test_healthbench():
    """Test the HealthBench dataset loader and evaluator"""
    
    print("=" * 80)
    print("Testing HealthBench Dataset Loader")
    print("=" * 80)
    
    # Load the dataset
    dataset_path = "datasets/healthbench/hard_2025-05-08-21-00-10.jsonl"
    loader = HealthBenchDatasetLoader(dataset_path=dataset_path, shuffle=False)
    
    print(f"\nDataset loaded successfully!")
    print(f"Total examples: {loader.get_total_count()}")
    print(f"Evaluator type: {loader.get_evaluator()}")
    print(f"Required tools: {loader.get_required_tools()}")
    
    # Get first example
    examples = loader.get_examples(1)
    example = examples[0]
    
    print(f"\n{'=' * 80}")
    print(f"Example ID: {example['id']}")
    print(f"{'=' * 80}")
    
    print(f"\nQuestion (first 500 chars):")
    print(example['question'][:500] + "...")
    
    print(f"\nMetadata:")
    print(f"  - Tags: {example['meta']['tags']}")
    print(f"  - Number of rubrics: {len(example['meta']['rubrics'])}")
    
    print(f"\nFirst 3 rubrics:")
    for i, rubric in enumerate(example['meta']['rubrics'][:3], 1):
        print(f"\n  Rubric {i}:")
        print(f"    Points: {rubric.get('points', 0)}")
        print(f"    Criterion: {rubric.get('criterion', '')[:200]}...")
    
    print(f"\n{'=' * 80}")
    print("Testing HealthBench Evaluator")
    print("=" * 80)
    
    # Initialize LLM client
    llm_client = LLMClientFactory.create_client()
    
    # Create evaluator
    evaluator = HealthBenchEvaluator()
    
    # Create a simple LLM caller function
    def llm_caller(prompt, system_instruction="You are a helpful assistant."):
        response = llm_client.generate(
            prompt=prompt,
            system_instruction=system_instruction
        )
        return response  # generate() already returns a string
    
    evaluator.set_llm_caller(llm_caller)
    
    # Create a mock system answer for testing
    mock_answer = """Based on the latest AHA 2020 Guidelines:

1. **Compression Depth**: The recommended compression depth remains at 5-6 cm (approximately 2-2.4 inches) for adults. This has not changed from previous guidelines.

2. **Epinephrine Dosing**: The standard dose remains 1 mg IV/IO every 3-5 minutes during cardiac arrest for both shockable and non-shockable rhythms.

3. **Advanced Airway Management**: Recent evidence suggests that there is no clear superiority between endotracheal intubation (ETI) and supraglottic airway devices (SGAs) in terms of patient outcomes. The key is to minimize interruptions in chest compressions. Either approach is considered reasonable (Class 2a recommendation). The choice should depend on provider expertise and the clinical context.

For non-shockable rhythms, epinephrine should be given as soon as possible. For shockable rhythms, it should be administered after initial defibrillation attempts.

Let me know if you need more specific details about any of these guidelines!"""
    
    print("\nEvaluating a mock response...")
    print(f"Mock answer (first 300 chars): {mock_answer[:300]}...")
    
    # Evaluate the mock answer
    evaluation = evaluator.evaluate(
        system_answer=mock_answer,
        golden_answer="",  # Not used in HealthBench
        context=example
    )
    
    print(f"\nEvaluation Results:")
    print(f"  - Match: {evaluation['match']}")
    print(f"  - Score: {evaluation['score']:.3f}")
    print(f"  - Confidence: {evaluation['confidence']:.3f}")
    print(f"  - Total Points: {evaluation['total_points']:.1f} / {evaluation['max_possible_points']:.1f}")
    print(f"  - Criteria Met: {evaluation['criteria_met']} / {evaluation['criteria_evaluated']}")
    print(f"  - Explanation: {evaluation['explanation']}")
    
    if evaluation['criteria_met_details']:
        print(f"\nFirst 3 criteria that were met:")
        for i, criterion in enumerate(evaluation['criteria_met_details'][:3], 1):
            print(f"\n  {i}. Points: {criterion['points']}")
            print(f"     Criterion: {criterion['criterion'][:150]}...")
    
    if evaluation['criteria_not_met_details']:
        print(f"\nFirst 3 criteria that were NOT met:")
        for i, criterion in enumerate(evaluation['criteria_not_met_details'][:3], 1):
            print(f"\n  {i}. Points: {criterion['points']}")
            print(f"     Criterion: {criterion['criterion'][:150]}...")
    
    print(f"\n{'=' * 80}")
    print("Test completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    try:
        test_healthbench()
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

