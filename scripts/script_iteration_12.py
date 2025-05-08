#!/usr/bin/env python
"""This script solves grid transformation problems by extracting, refining, and applying transformation rules,
with enhanced prompts and detailed reasoning steps."""

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

def rule_extraction(question: str) -> str:
    """Extract a transformation rule with an embedded example."""
    prompt = f"""You are an expert grid transformation expert.
    Analyze the provided question and extract the transformation rule in a structured format.
    Example:
    question: === TRAINING EXAMPLES === Example 1: Input Grid: [[1, 2], [3, 4]] Output Grid: [[4, 3], [2, 1]] === TEST INPUT === [[5, 6], [7, 8]] Transform the test input.
    Extracted Rule: {{"description": "The input grid is a 2x2 matrix of integers.","operations": "The matrix is flipped both horizontally and vertically.","output_description": "The output grid is the input grid flipped horizontally and vertically."}}
    question: {question}
    Extracted Rule:"""
    extracted_rule = call_llm(prompt)
    return extracted_rule

def refine_rule(question: str, extracted_rule: str) -> str:
  """Refine the rule, adding more specifics from the prompt."""
  prompt = f"""You are an expert grid transformation agent. Refine the Extracted Rule below.
  Extracted Rule: {extracted_rule}
  Here's the question for context: {question}
  Here is how a refined rule looks:
  Example:
    question: === TRAINING EXAMPLES === Example 1: Input Grid: [[1, 2], [3, 4]] Output Grid: [[4, 3], [2, 1]] === TEST INPUT === [[5, 6], [7, 8]] Transform the test input.
    Extracted Rule: {{"description": "The input grid is a 2x2 matrix of integers.","operations": "The matrix is flipped horizontally and vertically.","output_description": "The output grid is the input grid flipped horizontally and vertically."}}
    New Extracted Rule: {{"description": "The input grid is a 2x2 matrix.","operations": "The matrix is flipped. output[0][0] = input[1][1], output[0][1] = input[1][0], output[1][0] = input[0][1], and output[1][1] = input[0][0]","output_description": "The output grid is flipped."}}
  Refine the rule, if incorrect. Return the NEW Extracted Rule."""
  new_extracted_rule = call_llm(prompt)
  return new_extracted_rule

def apply_rule(input_grid: str, transformation_rule: str) -> str:
    """Apply the refined rule to the test input."""
    prompt = f"""You are an expert grid transformation agent. Apply the rule to the input_grid.
    input_grid: {input_grid}
    transformation_rule: {transformation_rule}
    Here is an example:
    transformation_rule: {{"description": "The input grid is a 2x2 matrix.","operations": "The matrix is flipped. output[0][0] = input[1][1], output[0][1] = input[1][0], output[1][0] = input[0][1], and output[1][1] = input[0][0]","output_description": "The output grid is flipped."}}
    input_grid: [[5, 6], [7, 8]]
    Output: [[8, 7], [6, 5]]
    Apply the rule to the grid and return it. Provide ONLY the grid."""
    transformed_grid = call_llm(prompt)
    return transformed_grid

def verify_grid(input_grid: str, transformed_grid: str, transformation_rule: str) -> str:
  """Verify that the transformed grid is valid based on the input grid and transformation rule."""
  prompt = f"""You are an expert grid transformation verifier. Verify that the transformed grid is valid based on the input grid and transformation rule.
  input_grid: {input_grid}
  transformed_grid: {transformed_grid}
  transformation_rule: {transformation_rule}
  Here is an example:
  input_grid: [[5, 6], [7, 8]]
  transformed_grid: [[8, 7], [6, 5]]
  transformation_rule: {{"description": "The input grid is a 2x2 matrix.","operations": "The matrix is flipped. output[0][0] = input[1][1], output[0][1] = input[1][0], output[1][0] = input[0][1], and output[1][1] = input[0][0]","output_description": "The output grid is flipped."}}
  Verification: The transformed grid is valid because it follows the transformation rule.
  Is the transformed grid valid? Explain why or why not."""
  verification = call_llm(prompt)
  return verification

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

        # 5. Verify the transformed grid
        verification = verify_grid(input_grid, transformed_grid, refined_rule)
        if "invalid" in verification.lower():
          return f"Error: Transformation produced an invalid result. {verification}"

        return transformed_grid
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred: {e}"