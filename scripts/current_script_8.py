#!/usr/bin/env python
"""
This script explores a new approach to grid transformation problems by using a decomposition and iterative refinement approach.

Hypothesis: By decomposing the grid transformation problem into identifying a subgrid that is transformed, then focusing on that subgrid and its transformation rules, and then iteratively
applying those transformations, the LLM can more accurately generalize the results. Additionally, iteratively refining through multiple calls allows for a better convergence.
"""

import os
import re
from typing import List, Dict, Any, Optional, Union

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response.  """
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

def find_transformable_subgrid(question: str) -> str:
    """
    Find the transformable subgrid in the question through LLM reasoning.
    """
    prompt = f"""
    You are an expert grid transformation expert.
    Determine what the transformable subgrid is.

    Example:
    question: 
    === TRAINING EXAMPLES ===
    Example 1:
    Input Grid: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    Output Grid: [[2, 3, 4], [5, 6, 7], [8, 9, 1]]
    === TEST INPUT ===
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    Transform the test input according to the pattern shown in the training examples.
    Output: "The transformable subgrid seems to be the entire grid, with each element being replaced by its immediate successor."

    question: {question}
    Output:
    """
    return call_llm(prompt)

def derive_subgrid_transformation_rule(question: str, subgrid_description: str) -> str:
    """
    Derive subgrid transformation rule in the question through LLM reasoning.
    """
    prompt = f"""
    You are an expert grid transformation expert.
    Analyze the following question and the subgrid. Infer the rules to derive the ouput grid.
    Example:
    question: 
    === TRAINING EXAMPLES ===
    Example 1:
    Input Grid: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    Output Grid: [[2, 3, 4], [5, 6, 7], [8, 9, 1]]
    === TEST INPUT ===
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    Transformable Subgrid: "The transformable subgrid seems to be the entire grid, with each element being replaced by its immediate successor."
    Output: "Each element is replaced by the immediate successor. For the element 9, wrap around so the successor is 1."

    question: {question}
    Transformable Subgrid: {subgrid_description}
    Output:
    """
    return call_llm(prompt)

def apply_transformation_to_grid(input_grid: str, transformation_rule: str) -> str:
    """Apply the transformation to the input grid. The input grid is a string"""
    prompt = f"""
    You are an expert grid transformation agent.
    Please transform the input_grid based on the transformation_rule.

    Example:
    input_grid: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    transformation_rule: Each element is replaced by the immediate successor. For the element 9, wrap around so the successor is 1.
    Output: "[[2, 3, 4], [5, 6, 7], [8, 9, 1]]"

    input_grid: {input_grid}
    transformation_rule: {transformation_rule}
    Output:
    """
    return call_llm(prompt)

def main(question: str) -> str:
    """Main function to solve the problem."""
    try:
        subgrid_description = find_transformable_subgrid(question)
        transformation_rule = derive_subgrid_transformation_rule(question, subgrid_description)
        
        test_input_match = re.search(r"=== TEST INPUT ===\n(.*?)\nTransform", question, re.DOTALL)
        if not test_input_match:
            return "Error: Could not find TEST INPUT in the question."
        input_grid = test_input_match.group(1).strip()

        transformed_grid = apply_transformation_to_grid(input_grid, transformation_rule)
        return transformed_grid

    except Exception as e:
        return f"An error occurred: {e}"