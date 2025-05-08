#!/usr/bin/env python
"""
This script explores a new approach to solving grid transformation problems by focusing on
identifying the core transformation logic through iterative refinement,
but doing so in a different and simple fashion. The problem is decomposed into
identifying the CORE logic and THEN testing this logic iteratively to refine.
The key is to keep the reasoning extremely simple, with steps.

Hypothesis: By focusing on an explicit simple transformation rule to solve, and
having the steps to confirm or deny the rule using each training grid, the rule will be generalized appropriately.
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

def identify_core_transformation_logic(question: str) -> str:
    """
    Identify the CORE transformation logic using a simple LLM call.
    This does not need to be correct, this is a hypothesis!
    """
    prompt = f"""
    You are a grid transformation expert.
    Given the question, identify the CORE transformation logic. Keep it very simple!
    Example:
    question:
    === TRAINING EXAMPLES ===
    Example 1:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[2, 3], [4, 1]]
    Transform the test input according to the pattern shown in the training examples.
    The CORE transformation logic is: Each number shifts to the right.

    question: {question}
    The CORE transformation logic is:
    """
    logic = call_llm(prompt)
    return logic

def verify_transformation_logic(question: str, logic: str) -> str:
    """
    Verify if all training examples fit with the same transformation logic.
    If ANY of the training examples do NOT fit this logic, explain why.
    """
    prompt = f"""
    You are a grid transformation expert.

    You have identified the CORE transformation logic as: {logic}

    Verify that ALL of the training examples adhere to this transformation logic.
    If ANY of the training examples do NOT fit this logic, explain why.
    Example:
    question:
    === TRAINING EXAMPLES ===
    Example 1:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[2, 3], [4, 1]]
    Transform the test input according to the pattern shown in the training examples.
    The CORE transformation logic is: Each number shifts to the right.
    Verification: Example 1 adheres to this transformation logic.

    question: {question}
    The CORE transformation logic is: {logic}
    Verification:
    """
    verification = call_llm(prompt)
    return verification

def apply_transformation_to_test_input(question: str, logic: str) -> str:
    """
    Apply the transformation logic to the test input, and return the resulting grid.
    """
    prompt = f"""
    You are a grid transformation expert.

    You have identified the CORE transformation logic as: {logic}

    Apply this to the test input and return the resulting grid as a list of lists.
    Example:
    Test Input: [[5, 6], [7, 8]]
    The CORE transformation logic is: Each number shifts to the right.
    Result: [[6, 7], [8, 5]]

    question: {question}
    The CORE transformation logic is: {logic}
    Result:
    """
    result = call_llm(prompt)
    return result

def main(question: str) -> str:
    """Main function to solve the problem."""
    try:
        # 1. Identify the core transformation logic
        core_logic = identify_core_transformation_logic(question)

        # 2. Verify if the logic applies to all examples.
        verification_result = verify_transformation_logic(question, core_logic)

        # 3. Check to see if the extraction is well formed and if it applies, or return an error.
        if "does NOT fit this logic" in verification_result:
            return f"Error: the identified CORE logic doesn't apply to all examples, because {verification_result}"
        else:
            print("Verification successful, can proceed with applying this logic.")

        # 4. Apply the transformation to the test input
        transformed_grid = apply_transformation_to_test_input(question, core_logic)
        return transformed_grid

    except Exception as e:
        return f"An error occurred: {e}"