#!/usr/bin/env python
"""
This script explores a new approach to grid transformation problems by using a 
combination of LLM-driven rule extraction with explicit positional reasoning and a verification loop.

Hypothesis: By explicitly representing positional information (row, col) in the LLM prompts and 
using a verification loop with feedback, we can improve the accuracy of rule extraction 
and application in grid transformation problems. This approach aims to address the LLM's
difficulty in reasoning about spatial relationships and ensure the transformations are applied 
consistently across the grid.
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

def solve_grid_transformation(question: str, max_attempts: int = 3) -> str:
    """Solve grid transformation using rule extraction with positional reasoning and verification."""
    # Step 1: Extract training examples and test input
    extraction_prompt = f"""
    Given this question, extract the training examples and the test input.

    {question}

    Format your response as follows:

    TRAINING_EXAMPLES:
    Example 1:
    Input Grid: [[...]]
    Output Grid: [[...]]
    Example 2:
    Input Grid: [[...]]
    Output Grid: [[...]]
    TEST_INPUT:
    [[...]]
    """
    extraction_result = call_llm(extraction_prompt)
    if "Error" in extraction_result:
        return "Error extracting information from the question."
    
    # Step 2: Extract transformation rule with positional reasoning
    rule_extraction_prompt = f"""
    Analyze the TRAINING_EXAMPLES below. Extract the transformation rule, 
    paying attention to how the value at each position (row, col) in the input grid 
    relates to the value at the same position in the output grid.

    Example:
    TRAINING_EXAMPLES:
    Example 1:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[2, 1], [4, 3]]
    Rule: The value at (row, col) is swapped with the value at (col, row).

    TRAINING_EXAMPLES:
    Example 1:
    Input Grid: [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    Output Grid: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    Rule: The value at input_grid[row][col] is copied to output_grid[col][row]

    {extraction_result}

    Provide the transformation rule, focusing on how the value at each (row, col) changes.
    """
    transformation_rule = call_llm(rule_extraction_prompt)
    if "Error" in transformation_rule:
        return "Error extracting transformation rule."
    
    # Step 3: Apply the transformation rule to the test input and perform verification
    application_prompt = f"""
    Apply the following transformation rule to the TEST_INPUT grid.

    Transformation Rule: {transformation_rule}

    TEST_INPUT:
    {extraction_result}

    Example:
    Transformation Rule: Each value gets incremented by 1
    TEST_INPUT: [[1,2],[3,4]]
    Output: [[2,3],[4,5]]

    Provide ONLY the transformed grid.
    """
    
    for attempt in range(max_attempts):
        application_result = call_llm(application_prompt)

        if "Error" in application_result:
            return "Error applying the transformation rule."
    
        # Step 4: Verify the output and provide feedback for refinement (Verification Loop)
        verification_prompt = f"""
        You are a grid transformation expert. You have applied the following
        transformation rule to the following TEST_INPUT and produced a result. 
        Verify if the result follows the stated rule.

        Transformation Rule: {transformation_rule}
        TEST_INPUT:
        {extraction_result}

        RESULT:
        {application_result}
        
        Example:
        Transformation Rule: The value at (row, col) is swapped with the value at (col, row).
        TEST_INPUT: [[1, 2], [3, 4]]
        RESULT: [[2, 1], [4, 3]]
        Verification: The result appears to be correct

        Determine if the RESULT matches the rule. If it does not match, point out what is wrong with the rule or the application
        
        Respond ONLY with "CORRECT" or "INCORRECT: [explain why the application failed and suggest how to fix it]"
        """
        verification_result = call_llm(verification_prompt)
        if "CORRECT" in verification_result:
            return application_result
        else:
            transformation_rule += f"\n REFINEMENT: {verification_result}" # Refine the rule by adding the issues to the rule
            print(f"Iteration {attempt + 1} failed. Reason: {verification_result}. Retrying...")

    return "Error occurred during processing after multiple attempts."

def main(question: str) -> str:
    """Main function to solve the problem."""
    answer = solve_grid_transformation(question)
    return answer