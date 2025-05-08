#!/usr/bin/env python
"""This script explores a new approach to solving grid transformation problems by using a pattern-based identification and transformation strategy. This contrasts previous approaches that focus on LLM rule extraction.

This approach differs from previous ones by:

1.  Identifying and categorizing common grid transformation patterns (shift, rotate, fill, etc).
2.  Using targeted prompting to identify the appropriate pattern.
3.  Applying LLM based rule transformation with an LLM to carry out the pattern transformation based on previous context.
4.  Using function test and verification steps to ensure a good output by checking against the examples.
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

def identify_transformation_type(question: str) -> str:
    """Identifies the type of transformation required."""
    prompt = f"""You are an expert in recognizing grid transformation patterns.
    Identify the primary transformation type in the following question.

    Example:
    question: === TRAINING EXAMPLES === Example 1: Input Grid: [[1, 2], [3, 4]] Output Grid: [[2, 1], [4, 3]] === TEST INPUT === [[5, 6], [7, 8]] Transform the test input.
    Transformation Type: Horizontal Flip

    question: {question}
    Transformation Type:"""
    transformation_type = call_llm(prompt)
    return transformation_type

def apply_transformation(input_grid: str, transformation_type: str, question: str) -> str:
    """Applies the identified transformation to the input grid."""
    prompt = f"""You are an expert in applying grid transformations.
    Apply the {transformation_type} transformation to the input grid.

    Example:
    input_grid: [[5, 6], [7, 8]]
    transformation_type: Horizontal Flip
    Transformed Grid: [[6, 5], [8, 7]]

    input_grid: {input_grid}
    transformation_type: {transformation_type}
    Question: {question}
    Transformed Grid:"""
    transformed_grid = call_llm(prompt)
    return transformed_grid

def function_test(input_grid: str, transformed_grid: str, question: str) -> str:
    """Test function to test transformation."""
    prompt = f"""You are an grid transformation expert. Test the new grid to make sure that the pattern has been successfully applied based on the question provided and the transformed grid that was made.
    Example of a successful function test, with explanation.
        question:
            === TRAINING EXAMPLES ===
            Example 1:
                Input Grid: [[1, 2], [3, 4]]
                Output Grid: [[2, 1], [4, 3]]
            === TEST INPUT ===
            [[5, 6], [7, 8]]
            Transform the test input according to the pattern shown in the training examples.
        transformed_grid: [[6, 5], [8, 7]]
    Result: [[6, 5], [8, 7]]
    The new grid displays a successful test because the columns swapped successfully based on the question provided.

    Example of an unsuccesful function test, with explanation.
        question:
            === TRAINING EXAMPLES ===
            Example 1:
                Input Grid: [[1, 2], [3, 4]]
                Output Grid: [[2, 1], [4, 3]]
            === TEST INPUT ===
            [[5, 6], [7, 8]]
            Transform the test input according to the pattern shown in the training examples.
        transformed_grid: [[5, 6], [7, 8]]
    Result: [[5, 6], [7, 8]]
    The new grid displays a failed test because the transformation was not applied.

    question: {question}
    transformation: {transformed_grid}
    Result: 
    """
    result = call_llm(prompt)
    return result

def main(question: str) -> str:
    """Main function to solve the problem."""
    try:
        # 1. Identify the transformation type
        transformation_type = identify_transformation_type(question)

        # 2. Extract the test input grid
        test_input_match = re.search(r"=== TEST INPUT ===\n(.*?)\nTransform", question, re.DOTALL)
        if not test_input_match:
            return "Error: Could not find TEST INPUT in the question."
        input_grid = test_input_match.group(1).strip()

        # 3. Apply the transformation
        transformed_grid = apply_transformation(input_grid, transformation_type, question)

        # 4. Apply function test to ensure the transformation occured successfully
        function_test_result = function_test(input_grid, transformed_grid, question)

        if "failed" in function_test_result:
            return f"Error: Function test has failed. {function_test_result}"

        return transformed_grid
    except Exception as e:
        return f"An error occurred: {e}"