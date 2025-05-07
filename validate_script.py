#!/usr/bin/env python
"""
simple_validator.py - A minimal script validator for ARC dataset

python validate_script.py --script scripts/script_iteration_21.py
"""
import os
import sys
import json
import importlib.util
import argparse
from pathlib import Path
from dataset_loader import create_dataset_loader
from google import genai
from google.genai import types

def compare_answers_with_llm(question, expected_answer, actual_answer):
    """
    Use LLM to compare if expected and actual answers are semantically equivalent,
    regardless of formatting differences.
    """
    # Initialize the Gemini client
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    prompt = f"""
    You are evaluating if two answers to a grid transformation problem are semantically equivalent.

    Original Question:
    {question}

    Expected Answer:
    {expected_answer}

    Actual Answer:
    {actual_answer}

    Are these answers semantically equivalent? Ignore formatting differences, variable names, 
    whitespace, and other non-semantic differences. Focus only on whether they represent the 
    same transformed grid or solution.

    Answer with ONLY 'YES' if they are equivalent or 'NO' if they are not.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=prompt
        )

        answer = response.text.strip().upper()
        return answer.startswith("YES")
    except Exception as e:
        print(f"Error comparing answers with LLM: {e}")
        # Fall back to string comparison in case of LLM error
        return str(expected_answer).strip() == str(actual_answer).strip()

def main():
    parser = argparse.ArgumentParser(description="Simple validator for ARC problems")
    parser.add_argument("--script", "-s", required=True, help="Path to script to validate")
    parser.add_argument("--last", "-l", type=int, default=50, help="Number of last examples to test (default: 50)")
    parser.add_argument("--dataset", "-d", type=str, default="ARC_2024_Training/", help="Path to ARC dataset")
    parser.add_argument("--detailed", "-D", action="store_true", help="Show detailed results")
    parser.add_argument("--exact-match", "-e", action="store_true", help="Use exact string matching only (no LLM comparison)")
    args = parser.parse_args()

    # Check API key
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set")
        sys.exit(1)

    # Load the script
    try:
        script_path = args.script
        print(f"Loading script from {script_path}")

        # Import the script as a module
        spec = importlib.util.spec_from_file_location("script_module", script_path)
        script_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script_module)

        if not hasattr(script_module, "main"):
            print("Error: Script does not have a main() function")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading script: {e}")
        sys.exit(1)

    # Create the dataset loader
    try:
        dataset_loader = create_dataset_loader("arc", dataset_path=args.dataset, shuffle=False)
        total_examples = dataset_loader.get_total_count()
        print(f"Loaded dataset with {total_examples} examples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # Calculate which examples to test
    start_idx = max(0, (total_examples-100) - args.last)
    examples_to_test = args.last
    print(f"Testing last {examples_to_test} examples (indices {start_idx} to {start_idx + examples_to_test - 1})")

    # Position the dataset loader
    dataset_loader.current_index = start_idx

    # Run validation
    correct = 0
    results = []

    for i in range(examples_to_test):
        try:
            # Get example
            examples = dataset_loader.get_examples(1)
            if not examples:
                print(f"Warning: Only got {i} examples (expected {examples_to_test})")
                break

            example = examples[0]
            question = dataset_loader.get_example_input(example)
            expected_answer = dataset_loader.get_example_output(example)

            # Print progress
            if i % 5 == 0:
                print(f"Testing example {i+1}/{examples_to_test}...")

            # Run the script
            try:
                actual_answer = script_module.main(question)
                success = True
                error = None
            except Exception as e:
                actual_answer = None
                success = False
                error = str(e)

            # Compare answers 
            if success:
                # First try exact string match for efficiency
                expected_str = str(expected_answer).strip()
                actual_str = str(actual_answer).strip()
                exact_match = expected_str == actual_str

                if exact_match:
                    is_correct = True
                    correct += 1
                elif not args.exact_match:
                    # Use LLM to compare for semantic equivalence if not using exact match only
                    print ("===========================================================================")
                    print("\n\n\nANSWER COMPARISON:\n\n\nSYSTEM ANSWER:\n\n\n", actual_answer, "\n\n\nGOLDEN ANSWER:\n\n\n", expected_answer, "\n\n\n")
                    is_correct = compare_answers_with_llm(question, expected_answer, actual_answer)
                    if is_correct:
                        correct += 1
                        print(f"  ✅ LLM determined answers are equivalent for example {start_idx + i}")
                        print ("===========================================================================")
                    
                else:
                    is_correct = False
                    print("  ❌ Answers do not match exactly")
                    print ("===========================================================================")
            else:
                is_correct = False

            # Save result
            results.append({
                "index": start_idx + i,
                "success": success,
                "correct": is_correct,
                "expected": expected_answer,
                "actual": actual_answer,
                "error": error
            })

        except Exception as e:
            print(f"Error processing example {start_idx + i}: {e}")

    # Calculate accuracy
    accuracy = correct / len(results) if results else 0

    # Print results
    print("\n=== Results ===")
    print(f"Tested examples: {len(results)}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {accuracy:.2f}")

    # Print detailed results if requested
    if args.detailed:
        print("\n=== Detailed Results ===")
        for result in results:
            status = "✅" if result["correct"] else "❌" if result["success"] else "⚠️"
            print(f"{status} Example {result['index']}: {'Success' if result['success'] else 'Error'}, {'Correct' if result['correct'] else 'Incorrect'}")
            if not result["success"] and result["error"]:
                print(f"   Error: {result['error']}")

if __name__ == "__main__":
    main()