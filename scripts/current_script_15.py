#!/usr/bin/env python
"""
This script improves grid transformation by:
1. Enhancing rule extraction/refinement with multi-example prompts and feedback.
2. Implementing a verification loop to ensure the refined rule is accurate.
3. Adding detailed comments and error handling.
"""

import os
import re
from typing import List, Dict, Any, Optional, Union

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt. DO NOT modify."""
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
    """Extract a transformation rule in structured format using LLM reasoning."""
    prompt = f"""
    You are an expert grid transformation analyst. Extract transformation rules with detailed descriptions of input, operations, and output.
    Example:
    question:
    === TRAINING EXAMPLES ===
    Example 1:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[4, 3], [2, 1]]
    Example 2:
    Input Grid: [[5, 6], [7, 8]]
    Output Grid: [[8, 7], [6, 5]]
    === TEST INPUT ===
    [[9, 10], [11, 12]]
    Transform the test input according to the pattern shown in the training examples.

    Extracted Rule:
    {{
      "description": "The input is a 2x2 matrix.",
      "operations": "Flip horizontally and vertically. output[i][j] = input[1-i][1-j]",
      "output_description": "Output is the input grid flipped."
    }}

    question: {question}
    Extracted Rule:
    """
    extracted_rule = call_llm(prompt)
    return extracted_rule

def refine_rule(question: str, extracted_rule: str, max_attempts=3) -> str:
    """Refine the extracted rule with a verification loop."""
    refined_rule = extracted_rule
    for attempt in range(max_attempts):
        prompt = f"""
        You are an expert grid transformation agent refining rules.

        Question: {question}
        Current Rule: {refined_rule}

        Example of Refinement:
        Question:
        === TRAINING EXAMPLES ===
        Example 1:
        Input Grid: [[1, 2], [3, 4]]
        Output Grid: [[4, 3], [2, 1]]
        === TEST INPUT ===
        [[5, 6], [7, 8]]
        Transform the test input according to the pattern shown in the training examples.
        Current Rule: {{"description": "2x2 matrix", "operations": "flip", "output_description": "flipped"}}
        Refined Rule: {{"description": "2x2 matrix", "operations": "flip horizontally and vertically", "output_description": "flipped horizontally and vertically"}}

        Refine the rule if incorrect, giving clear steps, but do NOT fix the output for the test example! Provide only a JSON rule.
        """
        new_refined_rule = call_llm(prompt)
        # Verification step:
        verification_prompt = f"""
        You are an expert grid transformation verifier.
        Verify the following refined rule against the original question.

        Question: {question}
        Refined Rule: {new_refined_rule}

        Does this rule appear complete and correct? Answer "Yes" or "No", followed by an explanation.
        """
        verification = call_llm(verification_prompt)
        if "Yes" in verification:
            return new_refined_rule
        else:
            refined_rule = new_refined_rule # try again with the updated rule.
    return refined_rule # Return the best effort after max attempts

def apply_rule(input_grid: str, transformation_rule: str) -> str:
    """Apply the refined transformation rule to the test input."""
    prompt = f"""
    You are an expert grid transformation agent.
    Apply the following transformation rule to the input grid.

    Input grid: {input_grid}
    Transformation rule: {transformation_rule}

    Example Application:
    Transformation rule: {{"description": "2x2 matrix", "operations": "flip horizontally and vertically", "output_description": "flipped horizontally and vertically"}}
    Input grid: [[5, 6], [7, 8]]
    Output: [[8, 7], [6, 5]]

    Apply the rule and return ONLY the transformed grid. Show your reasoning!
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