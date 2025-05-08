#!/usr/bin/env python
"""
This script refines a previous approach to solving grid transformation problems by enhancing the structured rule extraction and refinement process.
It integrates a verification loop with feedback for rule refinement and uses more detailed prompts with examples.
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

def rule_extraction(question: str) -> str:
    """Extract a transformation rule in structured format using LLM reasoning."""
    prompt = f"""
    You are an expert grid transformation analyst. Analyze the question and extract the transformation rule in a structured format.

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
      "operations": "The matrix is flipped horizontally and vertically.",
      "output_description": "The output grid is the input grid flipped horizontally and vertically."
    }}

    question: {question}
    Extracted Rule:
    """
    extracted_rule = call_llm(prompt, system_instruction="You are an expert at extracting grid transformation rules.")
    return extracted_rule

def refine_rule(question: str, extracted_rule: str, max_attempts=3) -> str:
    """Refine the extracted rule with a verification loop."""
    refined_rule = extracted_rule
    for attempt in range(max_attempts):
        verification_prompt = f"""
        You are a rule refinement expert. Here is the question: {question} and the extracted rule: {refined_rule}.

        Example:
        Question: === TRAINING EXAMPLES === Example 1: Input Grid: [[1, 2], [3, 4]] Output Grid: [[4, 3], [2, 1]] === TEST INPUT === [[5, 6], [7, 8]] Transform the test input.
        Extracted Rule: {{ "description": "2x2 matrix", "operations": "flip", "output_description": "flipped" }}
        Verification: The rule is too general. It needs to specify the flip directions.

        Refine: If the verification step above says the rule is incomplete, rewrite the Extracted Rule so that it accurately describes how to transform the input to the output. If it seems complete, then simply return the original rule.

        Please give a reason if the verification indicates the extracted rule can be more specific, and re-state the transformed rule with new specific instructions that help describe the correct transformation that is being applied between the input and output grids.
        """
        verification_result = call_llm(verification_prompt, system_instruction="You are an expert at verifying and refining rules.")
        if "The rule is too general" not in verification_result:  # Simple check for completeness
            break
        refined_rule = verification_result # This is an attempt to refine the prompt
    return refined_rule

def apply_rule(input_grid: str, transformation_rule: str) -> str:
    """Apply the refined transformation rule to the test input."""
    prompt = f"""
    You are an expert grid transformation agent. Apply the rule to the input_grid.

    Example:
    transformation_rule:
    {{
      "description": "The input grid is a 2x2 matrix of integers.",
      "operations": "The matrix is flipped both horizontally and vertically. Specifically, output[0][0] = input[1][1], output[0][1] = input[1][0], output[1][0] = input[0][1], and output[1][1] = input[0][0]",
      "output_description": "The output grid is the input grid flipped horizontally and vertically."
    }}
    input_grid: [[5, 6], [7, 8]]
    Output: [[8, 7], [6, 5]]

    input_grid: {input_grid}
    transformation_rule: {transformation_rule}

    Apply the rule to the grid and return it. Provide ONLY the grid.
    """
    transformed_grid = call_llm(prompt, system_instruction="You are an expert at applying rules to input grids.")
    return transformed_grid

def main(question: str) -> str:
    """Main function to solve the problem."""
    try:
        # 1. Extract the transformation rule
        extracted_rule = rule_extraction(question)

        # 2. Refine the transformation rule
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