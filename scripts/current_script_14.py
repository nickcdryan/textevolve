#!/usr/bin/env python
"""
This script refines the approach to solving grid transformation problems by incorporating stronger examples and detailed instructions for the LLM.
It builds upon the structured rule extraction and iterative refinement strategy of iteration 9 while addressing weaknesses.
"""

import os
import re
from typing import List, Dict, Any, Optional, Union

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response."""
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

def rule_extraction(question: str) -> str:
    """Extract a transformation rule in structured format using LLM reasoning."""
    prompt = f"""
    You are an expert grid transformation expert.

    Analyze the provided question and extract the transformation rule in a structured format.
    The structured rule should contain a description of the input, the operations being performed, and a description of the output. Focus on spatial relationships and dependencies.

    Example:
    question:
    === TRAINING EXAMPLES ===
    Example 1:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[4, 3], [2, 1]]
    === TEST INPUT ===
    [[5, 6], [7, 8]]
    Transform the test input according to the pattern shown in the training examples.

    Extracted Rule:
    {{
      "description": "The input grid is a 2x2 matrix of integers.",
      "operations": "The matrix is flipped both horizontally and vertically.  output[0][0] becomes input[1][1], output[0][1] becomes input[1][0], output[1][0] becomes input[0][1], and output[1][1] becomes input[0][0]",
      "output_description": "The output grid is the input grid flipped horizontally and vertically."
    }}

    question: {question}
    Extracted Rule:
    """
    extracted_rule = call_llm(prompt)
    return extracted_rule

def refine_rule(question: str, extracted_rule: str) -> str:
  """Refine the extracted rule if its incorrect."""
  prompt = f"""
  You are an expert grid transformation agent. You must analyze and refine the following Extracted Rule:
  {extracted_rule}

  Here is the question, so you can help to validate the rule:
  question: {question}

  Here is how it should perform, using the same question format:
  Example:
    question:
    === TRAINING EXAMPLES ===
    Example 1:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[4, 3], [2, 1]]
    === TEST INPUT ===
    [[5, 6], [7, 8]]
    Transform the test input according to the pattern shown in the training examples.

  Extracted Rule:
    {{
      "description": "The input grid is a 2x2 matrix of integers.",
      "operations": "The matrix is flipped both horizontally and vertically.",
      "output_description": "The output grid is the input grid flipped horizontally and vertically."
    }}

    New Extracted Rule:
    {{
      "description": "The input grid is a 2x2 matrix of integers.",
      "operations": "The matrix is flipped both horizontally and vertically. Specifically, output[0][0] = input[1][1], output[0][1] = input[1][0], output[1][0] = input[0][1], and output[1][1] = input[0][0]",
      "output_description": "The output grid is the input grid flipped horizontally and vertically."
    }}

  Please refine the rule, if its incorrect. Be as specific as possible about which parts of the grid change and what dependencies there are. If the rule looks correct, just return the rule as is, with no edits.
  """
  new_extracted_rule = call_llm(prompt)
  return new_extracted_rule

def apply_rule(input_grid: str, transformation_rule: str) -> str:
    """Apply the refined transformation rule to the test input."""
    prompt = f"""
    You are an expert grid transformation agent.
    You are given the following input grid, and a description about how to transform it. Apply the rule to the input_grid.

    input_grid: {input_grid}
    transformation_rule: {transformation_rule}

    Here is an example:
    transformation_rule:
    {{
      "description": "The input grid is a 2x2 matrix of integers.",
      "operations": "The matrix is flipped both horizontally and vertically. Specifically, output[0][0] = input[1][1], output[0][1] = input[1][0], output[1][0] = input[0][1], and output[1][1] = input[0][0]",
      "output_description": "The output grid is the input grid flipped horizontally and vertically."
    }}
    input_grid: [[5, 6], [7, 8]]
    Output: [[8, 7], [6, 5]]

    Apply the rule to the grid and return it. Provide ONLY the grid. Make sure that it is a valid grid, that each element is an integer, and that the formatting is correct. Be extremely precise and careful in applying the transformation.
    """
    transformed_grid = call_llm(prompt)
    return transformed_grid

def main(question: str) -> str:
    """Main function to solve the problem."""
    try:
        # 1. Extract the transformation rule
        extracted_rule = rule_extraction(question)

        # 2. Refine the transformation rule, to attempt to correct errors
        refined_rule = refine_rule(question, extracted_rule)

        # 3. Extract the test input grid
        test_input_match = re.search(r"=== TEST INPUT ===\n(.*?)\nTransform", question, re.DOTALL)
        if not test_input_match:
            return "Error: Could not find TEST INPUT in the question."
        input_grid = test_input_match.group(1).strip()

        # 4. Apply the refined transformation rule to the test input grid
        transformed_grid = apply_rule(input_grid, refined_rule)

        return transformed_grid
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred: {e}"