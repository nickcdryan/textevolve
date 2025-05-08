#!/usr/bin/env python
"""
Exploration: Meta-Reasoning with Iterative Pattern Refinement.
Hypothesis: A meta-reasoning agent that iteratively analyzes and refines its understanding of patterns will improve performance.

This approach differs significantly from previous ones by:
1. Introducing a meta-reasoning layer that reflects on and improves the transformation rule.
2. Breaking the problem into meta-analysis, rule application, and verification.
3. Minimizing the use of code, and maximizing the use of LLMs.
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

def meta_analyze_patterns(question: str, max_attempts=3) -> str:
    """Analyzes training examples and suggests a transformation pattern."""
    prompt = f"""You are a meta-reasoning agent, skilled at understanding complex patterns.
    Analyze the question and identify potential transformation patterns.

    Example:
    question: === TRAINING EXAMPLES === Example 1: Input Grid: [[1, 2], [3, 4]] Output Grid: [[4, 3], [2, 1]] === TEST INPUT === [[5, 6], [7, 8]] Transform the test input.
    Transformation Patterns: The grid is flipped horizontally and vertically. Values are swapped across diagonals.

    question: {question}
    Transformation Patterns:"""
    transformation_patterns = call_llm(prompt)
    return transformation_patterns

def apply_transformation(input_grid: str, transformation_patterns: str) -> str:
    """Applies the transformation to the input grid."""
    prompt = f"""You are a transformation agent, ready to apply patterns.
    Apply the transformation patterns to the input grid.

    Example:
    input_grid: [[5, 6], [7, 8]]
    transformation_patterns: The grid is flipped horizontally and vertically.
    Transformed Grid: [[8, 7], [6, 5]]

    input_grid: {input_grid}
    transformation_patterns: {transformation_patterns}
    Transformed Grid:"""
    transformed_grid = call_llm(prompt)
    return transformed_grid

def verify_transformation(question: str, transformed_grid: str) -> str:
    """Verifies if the transformation is correct."""
    prompt = f"""You are a verification agent, making sure transformations are correct.
    Verify if the transformation is correct based on the training examples in the question.

    Example:
    question: === TRAINING EXAMPLES === Example 1: Input Grid: [[1, 2], [3, 4]] Output Grid: [[2, 1], [4, 3]] === TEST INPUT === [[5, 6], [7, 8]] Transform the test input.
    transformed_grid: [[6, 5], [8, 7]]
    Is the transformation correct? Yes, the columns were swapped.

    question: {question}
    transformed_grid: {transformed_grid}
    Is the transformation correct?"""
    is_correct = call_llm(prompt)
    return is_correct

def main(question: str) -> str:
    """Main function to solve the problem."""
    try:
        # 1. Extract the test input grid
        test_input_match = re.search(r"=== TEST INPUT ===\n(.*?)\nTransform", question, re.DOTALL)
        if not test_input_match:
            return "Error: Could not find TEST INPUT in the question."
        input_grid = test_input_match.group(1).strip()

        # 2. Meta-analyze patterns
        transformation_patterns = meta_analyze_patterns(question)

        # 3. Apply the transformation
        transformed_grid = apply_transformation(input_grid, transformation_patterns)

        # 4. Verify the transformation
        is_correct = verify_transformation(question, transformed_grid)

        if "No" in is_correct:
            return "Transformation incorrect. Check the training examples"
        else:
            return transformed_grid
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred: {str(e)}"