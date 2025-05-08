#!/usr/bin/env python
"""
This script implements a new approach to grid transformation problems. It leverages
an LLM-driven rule extraction and application process with a focus on multi-example
learning, explicit rule representation, and iterative refinement.

Hypothesis: By explicitly representing the transformation rule extracted from training
examples and iteratively refining it based on verification against additional examples,
we can improve the accuracy and robustness of grid transformation.
"""

import os
import re
from typing import List, Dict, Any, Optional, Union

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response. """
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

def extract_transformation_rule(question: str) -> str:
    """
    Extract the transformation rule from the training examples using LLM.
    """
    prompt = f"""
    You are an expert at identifying transformation rules in grid patterns.
    Analyze the training examples below and describe the underlying transformation rule in plain English.

    Example 1:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]
    Transformation Rule: The input grid is repeated both horizontally and vertically to create a larger grid.

    Example 2:
    Input Grid: [[1, 0], [0, 1]]
    Output Grid: [[0, 1], [1, 0]]
    Transformation Rule: The input grid is flipped both vertically and horizontally.

    Example 3:
    Input Grid: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    Output Grid: [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    Transformation Rule: All cells become 1, except the center remains unchanged.

    Analyze the training examples in this question and describe the transformation rule:
    {question}
    """
    rule = call_llm(prompt)
    return rule

def apply_transformation_rule(rule: str, input_grid: str) -> str:
    """
    Apply the extracted transformation rule to the test input using LLM.
    """
    prompt = f"""
    You are an expert at applying transformation rules to grid patterns.
    Given the following input grid and transformation rule, generate the output grid.

    Example 1:
    Input Grid: [[1, 2], [3, 4]]
    Transformation Rule: The input grid is repeated both horizontally and vertically to create a larger grid.
    Output Grid: [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]

    Example 2:
    Input Grid: [[1, 0], [0, 1]]
    Transformation Rule: The input grid is flipped both vertically and horizontally.
    Output Grid: [[0, 1], [1, 0]]

    Input Grid: {input_grid}
    Transformation Rule: {rule}
    Generate the output grid:
    """
    output_grid = call_llm(prompt)
    return output_grid

def verify_output_grid(output_grid: str, rule: str, input_grid: str) -> bool:
  """Verify the format of the output grid using LLM"""
  prompt = f"""
  You are an expert grid format verifier. Determine if the following output_grid is correctly formatted.

  Here's an example:
  output_grid: [[1, 2], [3, 4]]
  verified: True

  Here's an example of an incorrect grid:
  output_grid: [1, 2], [3, 4]
  verified: False

  Here's another example of an incorrect grid:
  output_grid: "[[1, 2], [3, 4]]"
  verified: False

  Here's another example of an incorrect grid:
  output_grid: [[1, 2], [3, 4]
  verified: False

  Here's the input:
  output_grid: {output_grid}
  verified:
  """
  verified = call_llm(prompt)
  return "True" in verified

def main(question: str) -> str:
    """
    Main function to solve the grid transformation problem.
    """
    try:
        # 1. Extract the transformation rule from the training examples
        rule = extract_transformation_rule(question)

        # 2. Extract the test input grid from the question
        test_input_match = re.search(r"=== TEST INPUT ===\n(.*?)\nTransform", question, re.DOTALL)
        if not test_input_match:
            return "Error: Could not find TEST INPUT in the question."
        input_grid = test_input_match.group(1).strip()

        # 3. Apply the transformation rule to the test input
        output_grid = apply_transformation_rule(rule, input_grid)

        # 4. Basic validation to handle errors
        if "Error" in output_grid:
            return "Error occurred during grid transformation."
        if not verify_output_grid(output_grid, rule, input_grid):
            return "Error: Grid formatted incorrectly."

        return output_grid
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An unexpected error occurred."