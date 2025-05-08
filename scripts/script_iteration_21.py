#!/usr/bin/env python
"""
Refines iteration 9 and 15 to solve grid transformation problems through structured rule extraction, refinement, and application.
Addresses primary failure modes: pattern extraction and generalization. Incorporates iterative refinement with specific feedback.
Uses direct LLM reasoning approach to minimize parsing errors. Employs chain-of-thought reasoning and robust error handling.
"""

import os
import re
from typing import List, Dict, Any, Optional, Union

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response. DO NOT deviate from this example template or invent configuration options. This is how you call the LLM."""
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
    """
    Extract a transformation rule in structured format using LLM reasoning.
    Includes an example to guide the LLM.
    """
    prompt = f"""
    You are an expert grid transformation expert. Analyze the provided question and extract the transformation rule.

    Example:
    question:
    === TRAINING EXAMPLES ===
    Example 1:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[4, 3], [2, 1]]
    === TEST INPUT ===
    [[5, 6], [7, 8]]
    Transform the test input according to the pattern shown in the training examples.

    Extracted Rule: The input grid is flipped horizontally and vertically. Specifically, output[0][0] = input[1][1], output[0][1] = input[1][0], output[1][0] = input[0][1], and output[1][1] = input[0][0].

    question: {question}
    Extracted Rule:
    """
    extracted_rule = call_llm(prompt)
    return extracted_rule

def refine_rule(question: str, extracted_rule: str) -> str:
  """Refine the extracted rule, to attempt to correct errors. Includes example."""
  prompt = f"""
  You are an expert grid transformation agent. Refine the following extracted rule: {extracted_rule}

  Example:
    question:
    === TRAINING EXAMPLES ===
    Example 1:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[4, 3], [2, 1]]
    === TEST INPUT ===
    [[5, 6], [7, 8]]
    Transform the test input according to the pattern shown in the training examples.

  Extracted Rule: The input grid is flipped horizontally and vertically.
  Refined Rule: The input grid is flipped horizontally and vertically. Specifically, output[0][0] = input[1][1], output[0][1] = input[1][0], output[1][0] = input[0][1], and output[1][1] = input[0][0].

  Refine the rule based on the question: {question}. Return the refined rule.
  """
  refined_rule = call_llm(prompt)
  return refined_rule

def apply_rule(input_grid: str, transformation_rule: str) -> str:
    """Apply the refined transformation rule to the test input. Includes example."""
    prompt = f"""
    You are an expert grid transformation agent. Apply the rule to the input_grid.

    input_grid: {input_grid}
    transformation_rule: {transformation_rule}

    Example:
    transformation_rule: The input grid is flipped horizontally and vertically. Specifically, output[0][0] = input[1][1], output[0][1] = input[1][0], output[1][0] = input[0][1], and output[1][1] = input[0][0].
    input_grid: [[5, 6], [7, 8]]
    Output: [[8, 7], [6, 5]]

    Apply the rule to the grid and return the transformed grid. Provide ONLY the grid.
    """
    transformed_grid = call_llm(prompt)
    return transformed_grid

def main(question: str) -> str:
    """Main function to solve the problem. Includes robust error handling."""
    try:
        # 1. Extract the transformation rule
        extracted_rule = rule_extraction(question)
        if "Error" in extracted_rule:
            return f"Rule Extraction Error: {extracted_rule}"

        # 2. Refine the transformation rule, to attempt to correct errors
        refined_rule = refine_rule(question, extracted_rule)
        if "Error" in refined_rule:
            return f"Rule Refinement Error: {refined_rule}"

        # 3. Extract the test input grid
        test_input_match = re.search(r"=== TEST INPUT ===\n(.*?)\nTransform", question, re.DOTALL)
        if not test_input_match:
            return "Error: Could not find TEST INPUT in the question."
        input_grid = test_input_match.group(1).strip()

        # 4. Apply the refined transformation rule to the test input grid
        transformed_grid = apply_rule(input_grid, refined_rule)
        if "Error" in transformed_grid:
            return f"Rule Application Error: {transformed_grid}"

        return transformed_grid
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred: {e}"