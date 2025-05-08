#!/usr/bin/env python
"""This script explores a novel approach to solving grid transformation problems by focusing on iterative hypothesis generation and validation using a chain-of-thought with explicit testing of each hypothesis.

This is different from previous approaches by:

1.  Using the training examples to GENERATE multiple potential hypotheses about the transformation rule. Previous systems primarily used rules that were derived from prompt engineering or direct information extraction. This approach will derive the rule by explicitly asking for multiple possibilities.
2.  TESTING these hypotheses systematically against the examples.
3.  Choosing the hypothesis that best fits ALL examples.
4.  Applying the selected and tested hypotheses to the test input to generate the transformed grid.

This approach is designed to improve robustness and generalization by explicitly exploring and validating different potential rules, rather than relying on a single, potentially flawed, initial extraction. This relies on a direct LLM reasoning approach.
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

def generate_hypotheses(question: str, num_hypotheses: int = 3) -> List[str]:
    """Generates multiple hypotheses about the transformation rule from the training examples."""
    prompt = f"""You are an expert at analyzing grid transformation problems.
    Analyze the training examples in the provided question and generate {num_hypotheses} different hypotheses about the transformation rule.
    Consider various possibilities, including: shifting elements, replicating patterns, value-based modifications, and spatial relationships.
    Focus on generating logically distinct and plausible hypotheses.

    Example:
    question: === TRAINING EXAMPLES === Example 1: Input Grid: [[1, 2], [3, 4]] Output Grid: [[4, 3], [2, 1]] === TEST INPUT === [[5, 6], [7, 8]] Transform the test input.
    Hypotheses:
    1. The grid is flipped horizontally and vertically.
    2. The grid is rotated 180 degrees.
    3. The value at position [i][j] is swapped with the value at position [1-i][1-j].

    question: {question}
    Hypotheses:
    """
    hypotheses = call_llm(prompt)
    # Splitting the response into individual hypotheses for later testing
    return hypotheses.split("\n")

def test_hypotheses(question: str, hypotheses: List[str]) -> Dict[str, bool]:
    """Tests each hypothesis against the training examples and returns a dictionary of results."""
    prompt = f"""You are an expert at verifying hypotheses about grid transformations.
    Test each of the following hypotheses against the training examples provided in the question.
    For each hypothesis, determine whether it correctly explains the transformation in ALL training examples.

    Example:
    question: === TRAINING EXAMPLES === Example 1: Input Grid: [[1, 2], [3, 4]] Output Grid: [[4, 3], [2, 1]] === TEST INPUT === [[5, 6], [7, 8]] Transform the test input.
    Hypotheses:
    1. The grid is flipped horizontally and vertically.
    2. The grid is rotated 180 degrees.
    3. The value at position [i][j] is swapped with the value at position [1-i][1-j].
    Results:
    1. The grid is flipped horizontally and vertically. - Correct
    2. The grid is rotated 180 degrees. - Correct
    3. The value at position [i][j] is swapped with the value at position [1-i][1-j]. - Correct

    question: {question}
    Hypotheses:
    {chr(10).join([f"{i+1}. {h}" for i, h in enumerate(hypotheses)])}
    Results:
    """
    results = call_llm(prompt)
    # Parsing to return a dictionary of results for easy access and testing.
    results_dict = {}
    for i, hypothesis in enumerate(hypotheses):
        results_dict[hypothesis] = "Correct" in results.split(str(i+1) + ".")[1].split("\n")[0]
    return results_dict

def apply_transformation(input_grid: str, transformation_rule: str) -> str:
    """Applies the transformation rule to the test input."""
    prompt = f"""You are an expert in applying grid transformations.
    Apply the following transformation rule to the input grid.

    Input grid: {input_grid}
    Transformation rule: {transformation_rule}

    Example Application:
    Transformation rule: The grid is flipped horizontally and vertically.
    Input grid: [[5, 6], [7, 8]]
    Output: [[8, 7], [6, 5]]

    Apply the rule and return ONLY the transformed grid.
    """
    transformed_grid = call_llm(prompt)
    return transformed_grid

def main(question: str) -> str:
    """Main function to solve the problem."""
    try:
        # 1. Generate hypotheses about the transformation rule
        hypotheses = generate_hypotheses(question)

        # 2. Test the hypotheses against the training examples
        results = test_hypotheses(question, hypotheses)

        # 3. Select the best hypothesis (the one that correctly explains all examples)
        best_hypothesis = None
        for hypothesis, correct in results.items():
            if correct:
                best_hypothesis = hypothesis
                break

        if not best_hypothesis:
            return "Error: No hypothesis could be validated from the set. Unable to generate a useful output."

        # 4. Extract the test input grid
        test_input_match = re.search(r"=== TEST INPUT ===\n(.*?)\nTransform", question, re.DOTALL)
        if not test_input_match:
            return "Error: Could not find TEST INPUT in the question."
        input_grid = test_input_match.group(1).strip()

        # 5. Apply the transformation rule to the test input grid
        transformed_grid = apply_transformation(input_grid, best_hypothesis)

        return transformed_grid
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred: {e}"