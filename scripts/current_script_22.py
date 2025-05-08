#!/usr/bin/env python
"""This script explores a new approach to solving grid transformation problems by focusing on identifying "anchor" values and their influence on neighboring cells. The hypothesis is that transformations are driven by key "anchor" values, and their proximity determines how other cells change. A neighborhood influence propagation technique will be employed.

This approach differs from previous ones by:

1. Focusing on "anchor" values: The script will find key values that are most frequent in the training examples and apply a transformation based on what happens to their neighborhood
2.  Influence Propagation: The script will find patterns between "anchor" values and how nearby cells change
3. Applying neighborhood change based on influence: A process to extract the test matrix and transform the neighborhood with the identified influence propagations

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

def identify_anchor_values(question: str) -> str:
    """Identifies anchor values from the training examples."""
    prompt = f"""You are an expert in identifying key values in grid transformations.
    Analyze the training examples in the following question to identify the most frequent values, or "anchor" values, that seem to drive the transformations.

    Example:
    question: === TRAINING EXAMPLES === Example 1: Input Grid: [[1, 2], [3, 4]] Output Grid: [[2, 3], [4, 1]] === TEST INPUT === [[5, 6], [7, 8]] Transform the test input.
    Anchor Values: 1, 2, 3, 4 (all values appear to be equally important).

	question: === TRAINING EXAMPLES === Example 1: Input Grid: [[0, 0], [0, 4]] Output Grid: [[4, 4], [4, 4]] === TEST INPUT === [[0, 0], [0, 0]] Transform the test input.
    Anchor Values: 4 (4 seems to propagate).

    question: {question}
    Anchor Values:"""
    anchor_values = call_llm(prompt)
    return anchor_values

def analyze_neighborhood_influence(question: str, anchor_values: str) -> str:
    """Analyzes how anchor values influence their neighboring cells."""
    prompt = f"""You are an expert at analyzing grid transformations.
    Analyze the training examples in the following question and determine how the identified anchor values influence their neighboring cells in the output grid.

    Example:
    question: === TRAINING EXAMPLES === Example 1: Input Grid: [[0, 0], [0, 4]] Output Grid: [[4, 4], [4, 4]] === TEST INPUT === [[5, 6], [7, 8]] Transform the test input.
    Anchor Values: 4
    Neighborhood Influence: The value '4' seems to propagate to all neighboring cells, replacing their original values.

    question: {question}
    Anchor Values: {anchor_values}
    Neighborhood Influence:"""
    neighborhood_influence = call_llm(prompt)
    return neighborhood_influence

def transform_grid(input_grid: str, anchor_values: str, neighborhood_influence: str) -> str:
    """Transforms the input grid based on anchor values and their neighborhood influence."""
    prompt = f"""You are an expert in applying grid transformations.
    Apply the transformation to the provided input grid, based on the anchor values and their influence on neighboring cells.

    Example:
    input_grid: [[5, 6], [7, 8]]
    anchor_values: 8
    neighborhood_influence: The value '8' seems to shift values left
    Transformed Grid: [[6, 5], [8, 7]]

    input_grid: {input_grid}
    anchor_values: {anchor_values}
    neighborhood_influence: {neighborhood_influence}
    Transformed Grid:"""
    transformed_grid = call_llm(prompt)
    return transformed_grid

def main(question: str) -> str:
    """Main function to solve the problem."""
    try:
        # 1. Identify anchor values
        anchor_values = identify_anchor_values(question)

        # 2. Analyze neighborhood influence
        neighborhood_influence = analyze_neighborhood_influence(question, anchor_values)

        # 3. Extract the test input grid
        test_input_match = re.search(r"=== TEST INPUT ===\n(.*?)\nTransform", question, re.DOTALL)
        if not test_input_match:
            return "Error: Could not find TEST INPUT in the question."
        input_grid = test_input_match.group(1).strip()

        # 4. Transform the grid
        transformed_grid = transform_grid(input_grid, anchor_values, neighborhood_influence)

        return transformed_grid
    except Exception as e:
        return f"An error occurred: {e}"