#!/usr/bin/env python
"""
Improved LLM-driven agent for solving grid transformation problems. This version enhances the direct pattern matching 
approach with multi-example prompting, verification, and targeted error handling.
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

def solve_grid_transformation(question: str, max_attempts: int = 3) -> str:
    """
    Solve grid transformation problems using direct pattern matching with LLM and verification.
    """
    for attempt in range(max_attempts):
        # Enhanced prompt with multiple examples and clear instructions
        prompt = f"""
        You are an expert at recognizing patterns in grid transformations. Given training examples
        and a test input, transform the test input according to the learned pattern.
        Pay close attention to how the training grids transform and the number of times that this transformation repeats.

        Example 1:
        Input Grid:
        [[0, 0, 8, 0, 0], [0, 0, 8, 0, 0], [8, 8, 8, 8, 8], [0, 0, 8, 0, 0], [0, 0, 8, 0, 0]]
        Output Grid:
        [[0, 0, 8, 0, 0], [0, 0, 8, 0, 0], [8, 8, 8, 8, 8], [0, 0, 8, 0, 0], [0, 0, 8, 0, 0]]

        Example 2:
        Input Grid:
        [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]
        Output Grid:
        [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]
        
        Example 3:
        Input Grid:
        [[2, 0, 3, 5, 4], [0, 0, 8, 5, 0], [4, 6, 0, 5, 3], [5, 5, 5, 5, 5], [4, 0, 8, 5, 0]]
        Output Grid:
        [[0, 0, 0, 5, 0], [0, 0, 0, 5, 0], [0, 0, 0, 5, 0], [5, 5, 5, 5, 5], [0, 0, 0, 5, 0]]

        Given the training examples and the TEST INPUT below, transform the TEST INPUT according to the patterns observed in the examples. Return ONLY the transformed grid.
        {question}
        """

        # Call the LLM
        llm_output = call_llm(prompt)

        # Implement basic validation: check if the output is non-empty and not an error message
        if llm_output and "Error" not in llm_output:
            return llm_output
        else:
            print(f"Attempt {attempt + 1} failed. Retrying...")

    return "Error occurred during processing after multiple attempts."

def main(question: str) -> str:
    """
    Main function to solve the grid transformation problem.
    """
    answer = solve_grid_transformation(question)
    return answer