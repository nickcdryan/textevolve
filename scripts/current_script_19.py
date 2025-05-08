#!/usr/bin/env python
"""
Refines the grid transformation approach focusing on structured rule extraction and refinement,
incorporating verification loops and detailed examples in LLM prompts.
"""

import os
import re
from typing import List, Dict, Any, Optional, Union

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response.
    DO NOT modify this or invent configuration options. This is how you call the LLM."""
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

def rule_extraction(question: str, max_attempts=3) -> str:
    """Extract transformation rule with example and verification."""
    prompt = f"""
    You are an expert grid transformation analyst. Extract a rule from the examples.

    Example:
    question: === TRAINING EXAMPLES === Example 1: Input Grid: [[1, 2], [3, 4]] Output Grid: [[4, 3], [2, 1]] === TEST INPUT === [[5, 6], [7, 8]] Transform the test input according to the pattern shown in the training examples.
    Extracted Rule: {{"description": "2x2 matrix", "operations": "flip horizontal and vertical", "output_description": "flipped matrix"}}

    question: {question}
    Extracted Rule:
    """
    for attempt in range(max_attempts):
        extracted_rule = call_llm(prompt)
        if extracted_rule:  # Basic check for non-empty response
            return extracted_rule
        print(f"Rule extraction failed, attempt {attempt + 1}/{max_attempts}")
    return "Error: Rule extraction failed after multiple attempts."

def refine_rule(question: str, extracted_rule: str, max_attempts=3) -> str:
    """Refine rule with example, verification and descriptive output."""
    prompt = f"""
    You are an expert at refining rules. Refine this rule based on examples from the question.

    Example:
    question: === TRAINING EXAMPLES === Example 1: Input Grid: [[1, 2], [3, 4]] Output Grid: [[4, 3], [2, 1]] === TEST INPUT === [[5, 6], [7, 8]] Transform the test input according to the pattern shown in the training examples.
    Extracted Rule: {{"description": "2x2 matrix", "operations": "flip horizontal and vertical", "output_description": "flipped matrix"}}
    Refined Rule: {{"description": "2x2 matrix", "operations": "output[0][0] = input[1][1], output[0][1] = input[1][0], output[1][0] = input[0][1], output[1][1] = input[0][0]", "output_description": "flipped matrix"}}

    question: {question}
    Extracted Rule: {extracted_rule}
    Refined Rule:
    """
    for attempt in range(max_attempts):
        refined_rule = call_llm(prompt)
        if refined_rule:
            return refined_rule
        print(f"Rule refinement failed, attempt {attempt + 1}/{max_attempts}")
    return "Error: Rule refinement failed after multiple attempts."

def apply_rule(input_grid: str, transformation_rule: str, max_attempts=3) -> str:
    """Apply rule with example and verification."""
    prompt = f"""
    Apply the rule to the input grid. You are an expert grid transformation agent.

    Example:
    transformation_rule: {{"description": "2x2 matrix", "operations": "output[0][0] = input[1][1], output[0][1] = input[1][0], output[1][0] = input[0][1], output[1][1] = input[0][0]", "output_description": "flipped matrix"}}
    input_grid: [[5, 6], [7, 8]]
    Output: [[8, 7], [6, 5]]

    input_grid: {input_grid}
    transformation_rule: {transformation_rule}
    Output:
    """
    for attempt in range(max_attempts):
        transformed_grid = call_llm(prompt)
        if transformed_grid:
            return transformed_grid
        print(f"Rule application failed, attempt {attempt + 1}/{max_attempts}")
    return "Error: Rule application failed after multiple attempts."

def main(question: str) -> str:
    """Main function with improved error handling."""
    try:
        extracted_rule = rule_extraction(question)
        if "Error:" in extracted_rule:
            return extracted_rule

        refined_rule = refine_rule(question, extracted_rule)
        if "Error:" in refined_rule:
            return refined_rule

        test_input_match = re.search(r"=== TEST INPUT ===\n(.*?)\nTransform", question, re.DOTALL)
        if not test_input_match:
            return "Error: Could not find TEST INPUT in the question."
        input_grid = test_input_match.group(1).strip()

        transformed_grid = apply_rule(input_grid, refined_rule)
        if "Error:" in transformed_grid:
            return transformed_grid

        return transformed_grid
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred: {str(e)}"