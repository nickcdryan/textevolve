#!/usr/bin/env python
"""
Improved LLM-driven agent for solving grid transformation problems. This version focuses on 
direct pattern matching, robust error handling, and minimal code.
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

def solve_grid_transformation(question: str) -> str:
    """
    Solve grid transformation problems using direct pattern matching with LLM.
    """
    # Simplified approach: Use the LLM to directly transform the input based on examples.
    prompt = f"""
    You are an expert at recognizing patterns in grid transformations. Given training examples
    and a test input, transform the test input according to the learned pattern.

    Example 1:
    Input Grid:
    [[0, 7, 7], [7, 7, 7], [0, 7, 7]]
    Output Grid:
    [[0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 0, 0, 7, 7, 7, 7, 7, 7], [0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 7, 7, 0, 7, 7, 0, 7, 7], [7, 7, 7, 7, 7, 7, 7, 7, 7], [0, 7, 7, 0, 7, 7, 0, 7, 7], [0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 0, 0, 7, 7, 7, 7, 7, 7], [0, 0, 0, 0, 7, 7, 0, 7, 7]]

    Example 2:
    Input Grid:
    [[4, 0, 4], [0, 0, 0], [0, 4, 0]]
    Output Grid:
    [[4, 0, 4, 0, 0, 0, 4, 0, 4], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 4, 0, 4, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 4, 0, 0, 0, 0]]

    Given the training examples and the TEST INPUT below, transform the TEST INPUT according to the patterns observed in the examples. Return ONLY the transformed grid.
    {question}
    """

    # Call the LLM
    llm_output = call_llm(prompt)

    # Implement very basic validation
    if "Error" in llm_output:
        return "Error occurred during processing."
    else:
        return llm_output

def main(question: str) -> str:
    """
    Main function to solve the grid transformation problem.
    """
    answer = solve_grid_transformation(question)
    return answer