#!/usr/bin/env python
"""
Exploration: Iterative Pattern Identification and Verification with Dynamic Example Selection

Hypothesis: Iteratively identifying and verifying transformation patterns, guided by a dynamic selection of relevant training examples, will improve grid transformation performance. This strategy aims to address the challenges of incorrect pattern deduction and limited generalization by focusing on robust pattern identification and validation before applying the transformation to the test input.

This approach differs significantly from previous ones by:

1.  Iterative Refinement of Transformation Patterns: Focuses on improving the accuracy of identified transformation patterns through iterative refinement and verification, rather than relying on a single extraction step.
2.  Dynamic Selection of Relevant Training Examples: Selects relevant training examples based on similarity to the test input, enabling the system to focus on the most relevant information for the transformation.
3.  Verification of Transformation Patterns: Verifies transformation patterns by applying them to training examples and comparing the results with the expected outputs.
4.  Multi-Agent Orchestration: Uses multiple LLM agents with specialized roles, including a pattern identifier, a pattern verifier, and a transformation applier.

"""

import os
import re
from typing import List, Dict, Any, Optional, Union

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        if system_instruction:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction
                ),
                contents=prompt
            )
        else:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )

        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return f"Error: {str(e)}"

def extract_training_and_test_data(question: str) -> Dict:
    """Extracts training examples and test input from the question."""
    prompt = f"""
    You are an expert at extracting structured data. Extract training examples and the test input.

    Example:
    question: === TRAINING EXAMPLES === Example 1: Input Grid: [[1, 2], [3, 4]] Output Grid: [[4, 3], [2, 1]] === TEST INPUT === [[5, 6], [7, 8]] Transform the test input.
    {{ "training_examples": [{{ "input": "[[1, 2], [3, 4]]", "output": "[[4, 3], [2, 1]]" }}], "test_input": "[[5, 6], [7, 8]]" }}

    question: {question}
    Extracted Data:
    """
    extracted_data = call_llm(prompt)
    try:
        # Avoid json.loads()
        return eval(extracted_data)
    except Exception as e:
        print(f"Error parsing extracted data: {e}")
        return {"training_examples": [], "test_input": ""}

def select_relevant_examples(training_examples: List[Dict], test_input: str) -> List[Dict]:
    """Selects the most relevant training examples based on similarity to the test input."""
    prompt = f"""
    You are an expert at selecting relevant training examples. Given the following training examples and test input, select the 2 most relevant examples based on structural similarity (grid size, density of non-zero elements).

    Example:
    training_examples: [{{ "input": "[[0, 0], [0, 1]]", "output": "[[1, 1], [1, 1]]" }}, {{ "input": "[[1, 1], [1, 0]]", "output": "[[0, 0], [0, 0]]" }}]
    test_input: "[[0, 1], [0, 0]]"
    Relevant Examples: [{{ "input": "[[0, 0], [0, 1]]", "output": "[[1, 1], [1, 1]]" }}, {{ "input": "[[1, 1], [1, 0]]", "output": "[[0, 0], [0, 0]]" }}]

    training_examples: {training_examples}
    test_input: {test_input}
    Relevant Examples:
    """
    relevant_examples_str = call_llm(prompt)
    try:
        # Avoid json.loads()
        relevant_examples = eval(relevant_examples_str)
        return relevant_examples
    except Exception as e:
        print(f"Error parsing relevant examples: {e}")
        return training_examples[:2] # Return first two if selection fails

def identify_transformation_pattern(relevant_examples: List[Dict]) -> str:
    """Identifies a transformation pattern from the relevant examples."""
    prompt = f"""
    You are an expert at identifying transformation patterns. Given the following relevant training examples, identify a general transformation pattern that explains all examples.

    Example:
    relevant_examples: [{{ "input": "[[1, 2], [3, 4]]", "output": "[[4, 3], [2, 1]]" }}]
    Transformation Pattern: The grid is flipped horizontally and vertically.

    relevant_examples: {relevant_examples}
    Transformation Pattern:
    """
    transformation_pattern = call_llm(prompt)
    return transformation_pattern

def verify_transformation_pattern(transformation_pattern: str, training_examples: List[Dict]) -> str:
    """Verifies the transformation pattern against the training examples."""
    prompt = f"""
    You are an expert at verifying transformation patterns. Given the following transformation pattern and training examples, verify if the pattern correctly transforms the input grids to the output grids.

    Example:
    transformation_pattern: The grid is flipped horizontally and vertically.
    training_examples: [{{ "input": "[[1, 2], [3, 4]]", "output": "[[4, 3], [2, 1]]" }}]
    Verification Result: The transformation pattern is correct.

    transformation_pattern: {transformation_pattern}
    training_examples: {training_examples}
    Verification Result:
    """
    verification_result = call_llm(prompt)
    return verification_result

def apply_transformation(test_input: str, transformation_pattern: str) -> str:
    """Applies the transformation pattern to the test input."""
    prompt = f"""
    You are an expert at applying transformation patterns. Apply the following transformation pattern to the test input.

    Example:
    transformation_pattern: The grid is flipped horizontally and vertically.
    test_input: "[[5, 6], [7, 8]]"
    Transformed Grid: [[8, 7], [6, 5]]

    transformation_pattern: {transformation_pattern}
    test_input: {test_input}
    Transformed Grid:
    """
    transformed_grid = call_llm(prompt)
    return transformed_grid

def main(question: str) -> str:
    """Main function to solve the problem."""
    try:
        # 1. Extract training examples and test input
        extracted_data = extract_training_and_test_data(question)
        training_examples = extracted_data["training_examples"]
        test_input = extracted_data["test_input"]

        # 2. Select relevant examples
        relevant_examples = select_relevant_examples(training_examples, test_input)

        # 3. Identify transformation pattern
        transformation_pattern = identify_transformation_pattern(relevant_examples)

        # 4. Verify transformation pattern
        verification_result = verify_transformation_pattern(transformation_pattern, training_examples)

        if "incorrect" in verification_result.lower():
            return "Transformation pattern is incorrect. Check the training examples."

        # 5. Apply the transformation pattern to the test input
        transformed_grid = apply_transformation(test_input, transformation_pattern)

        return transformed_grid

    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred: {e}"