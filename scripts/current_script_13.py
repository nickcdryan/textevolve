#!/usr/bin/env python
"""This script explores a new approach to solving grid transformation problems.

Hypothesis: By focusing on identifying minimal change regions and then interpolating patterns based on those regions, we can solve these problems with greater accuracy. This contrasts previous approaches that relied on global pattern matching or iterative refinement of a single rule.

This approach leverages two distinct LLM agents: a Minimal Change Identifier and a Pattern Interpolator. It also includes a new verification step.
"""

import os
import re
from typing import List, Dict, Any, Optional, Union

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response."""
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

def identify_minimal_change_regions(question: str) -> str:
    """Identifies regions with minimal change between input and output grids."""
    prompt = f"""You are a Minimal Change Identifier. Analyze the grid transformation question and identify regions where the transformation is minimal or non-existent. This is crucial for anchoring pattern interpolation.

    Example:
    question: === TRAINING EXAMPLES === Example 1: Input Grid: [[1, 2], [3, 4]] Output Grid: [[1, 3], [2, 4]] === TEST INPUT === [[5, 6], [7, 8]] Transform the test input.
    Minimal Change Regions: The diagonal elements (top-left and bottom-right) remain unchanged.

    question: {question}
    Minimal Change Regions:"""
    minimal_change = call_llm(prompt)
    return minimal_change

def interpolate_transformation_pattern(question: str, minimal_change: str) -> str:
    """Interpolates the transformation pattern based on minimal change regions."""
    prompt = f"""You are a Pattern Interpolator. Given the grid transformation question and the identified minimal change regions, interpolate the transformation pattern that explains the changes. Use minimal change regions as a stable base to infer changes elsewhere.

    Example:
    question: === TRAINING EXAMPLES === Example 1: Input Grid: [[1, 2], [3, 4]] Output Grid: [[1, 3], [2, 4]] === TEST INPUT === [[5, 6], [7, 8]] Transform the test input.
    Minimal Change Regions: The diagonal elements (top-left and bottom-right) remain unchanged.
    Transformation Pattern: The off-diagonal elements swap positions.

    question: {question}
    Minimal Change Regions: {minimal_change}
    Transformation Pattern:"""
    transformation_pattern = call_llm(prompt)
    return transformation_pattern

def apply_transformation(input_grid: str, transformation_pattern: str) -> str:
    """Apply the interpolated transformation pattern to the input grid."""
    prompt = f"""You are a Grid Transformer. Apply the transformation pattern to the input grid to generate the transformed grid.

    Example:
    input_grid: [[5, 6], [7, 8]]
    transformation_pattern: The off-diagonal elements swap positions.
    Transformed Grid: [[5, 7], [6, 8]]

    input_grid: {input_grid}
    transformation_pattern: {transformation_pattern}
    Transformed Grid:"""
    transformed_grid = call_llm(prompt)
    return transformed_grid

def verify_transformation(question: str, transformed_grid: str) -> str:
    """Verifies that the transformation is valid by performing error analysis."""
    prompt = f"""You are an expert grid transformation verifier. Verify that the new grid provided makes sense given the question.
    Here is how it should perform, using the same question format:
    Example of a valid transformation, with explanation.
        question:
            === TRAINING EXAMPLES ===
            Example 1:
                Input Grid: [[1, 2], [3, 4]]
                Output Grid: [[2, 3], [4, 1]]
            === TEST INPUT ===
            [[5, 6], [7, 8]]
            Transform the test input according to the pattern shown in the training examples.

    transformation: [[6, 7], [8, 5]]
    verified: CORRECT because numbers shift to the right.

    question: {question}
    transformation: {transformed_grid}
    verified: 
    """
    verified = call_llm(prompt)
    return verified

def main(question: str) -> str:
    """Main function to solve the problem."""
    try:
        # 1. Identify minimal change regions
        minimal_change = identify_minimal_change_regions(question)

        # 2. Interpolate transformation pattern
        transformation_pattern = interpolate_transformation_pattern(question, minimal_change)

        # 3. Extract the test input grid
        test_input_match = re.search(r"=== TEST INPUT ===\n(.*?)\nTransform", question, re.DOTALL)
        if not test_input_match:
            return "Error: Could not find TEST INPUT in the question."
        input_grid = test_input_match.group(1).strip()

        # 4. Apply the transformation
        transformed_grid = apply_transformation(input_grid, transformation_pattern)

        # 5. Verify
        verified = verify_transformation(question, transformed_grid)

        if "INCORRECT" in verified:
            return f"Error: Transformation verification failed. {verified}"

        return transformed_grid
    except Exception as e:
        return f"An error occurred: {e}"