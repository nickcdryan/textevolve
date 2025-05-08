#!/usr/bin/env python
"""
This script introduces a novel approach to grid transformation problems by
focusing on localized pattern analysis and cell-by-cell transformation.

Hypothesis: By analyzing the neighborhood of each cell in the input grid
and using the training examples to determine the corresponding output cell value,
we can improve the accuracy of grid transformations. This approach attempts
to mitigate the difficulty of extracting global transformation rules by
focusing on local relationships and applying them systematically. This also
includes a validation check partway through the pipeline to see if the transformation
rule appears to be well-formed
"""

import os
import re
from typing import List, Dict, Any, Optional, Union

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response. DO NOT modify this or invent configuration options. This is how you call the LLM."""
    try:
        from google import genai
        from google.genai import types

        # Initialize the Gemini client
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        # Call the API with system instruction if provided
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

def analyze_cell_transformation(question: str, row: int, col: int) -> str:
    """
    Analyze how a specific cell transforms based on its neighborhood and training examples.
    """
    prompt = f"""
    You are a grid transformation expert. Analyze the transformation of a specific cell
    in the input grid based on its neighborhood and the provided training examples.

    Example:
    Question:
    === TRAINING EXAMPLES ===
    Example 1:
    Input Grid: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    Output Grid: [[2, 3, 4], [5, 6, 7], [8, 9, 1]]
    === TEST INPUT ===
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    Analyze the transformation of cell (0, 0).

    Analysis: The value 1 at cell (0, 0) in the input grid becomes 2 in the output grid.
    This suggests a shift of the first row to the right, or that each number becomes its following number with 9 wrapping to 1.

    Question:
    {question}
    Analyze the transformation of cell ({row}, {col}).
    """
    analysis = call_llm(prompt)
    return analysis

def apply_cell_transformation(question: str, row: int, col: int, analysis: str) -> str:
    """
    Apply the learned transformation to determine the output value of a specific cell.
    """
    prompt = f"""
    You are a grid transformation expert. Given the analysis of cell transformation
    and the question, determine the output value of a specific cell.

    Question:
    {question}
    Analysis of cell ({row}, {col}):
    {analysis}

    What is the transformed value of cell ({row}, {col}) in the output grid?
    Example:
    Question:
    === TRAINING EXAMPLES ===
    Example 1:
    Input Grid: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    Output Grid: [[2, 3, 4], [5, 6, 7], [8, 9, 1]]
    === TEST INPUT ===
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    Analyze the transformation of cell (0, 0).

    Analysis of cell (0, 0): The value 1 at cell (0, 0) in the input grid becomes 2 in the output grid.
    This suggests each number becomes its following number with 9 wrapping to 1.
    Output: 2
    
    Output:
    """
    output_value = call_llm(prompt)
    return output_value

def check_rule_well_formed(question: str) -> str:
    """
    Check to see that a rule is well-formed and self consistent.
    """
    prompt = f"""
    You are a grid transformation expert.

    Analyze the following question. Determine a well-formed rule from the 
    training grids from the question. This includes identifying how the input
    grids are transformed to output grids.
    Provide a brief description of the transformation rule.
    {question}
    """
    rule = call_llm(prompt)
    return rule

def solve_grid_transformation(question: str) -> str:
    """
    Solve the grid transformation problem by analyzing and transforming each cell individually.
    """
    test_input_match = re.search(r"=== TEST INPUT ===\n(.*?)\nTransform", question, re.DOTALL)
    if not test_input_match:
        return "Error: Could not find TEST INPUT in the question."
    input_grid_str = test_input_match.group(1).strip()
    input_grid = [list(map(int, re.findall(r'\d+', row))) for row in re.findall(r'\[.*?\]', input_grid_str)]
    rows = len(input_grid)
    cols = len(input_grid[0])

    output_grid = []
    for row in range(rows):
        output_row = []
        for col in range(cols):
            # Analyze the transformation of the cell
            analysis = analyze_cell_transformation(question, row, col)
            # Apply the transformation to get the output value
            output_value = apply_cell_transformation(question, row, col, analysis)
            try:
                output_row.append(int(output_value))
            except ValueError:
                print(f"Error: Could not convert '{output_value}' to integer. Returning error.")
                return "Error: Invalid transformation."
        output_grid.append(output_row)
    return str(output_grid)

def main(question: str) -> str:
    """Main function to solve the problem."""
    # Implement check rule well formed
    well_formed = check_rule_well_formed(question)

    if "Error" in well_formed:
        return "Error: There is no valid rule extracted."
    else:
        print("Rule successfully extracted.")

    answer = solve_grid_transformation(question)
    return answer