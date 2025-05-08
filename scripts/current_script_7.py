#!/usr/bin/env python
"""
Improved LLM-driven agent for solving grid transformation problems. This version focuses on 
direct pattern matching with enhanced few-shot examples and structured output.
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
    Solve grid transformation problems using direct pattern matching with LLM and enhanced examples.
    """
    # Enhanced prompt with more detailed examples demonstrating step-by-step reasoning.
    prompt = f"""
    You are an expert at recognizing patterns in grid transformations. Given training examples
    and a test input, transform the test input according to the learned pattern. Focus on spatial relationships.
    The output should be a transformed grid, formatted as a list of lists.
    Here are multiple examples of grid transformations that showcase step-by-step transformation reasoning.

    Example 1:
    Input Grid:
    [[0, 7, 7], [7, 7, 7], [0, 7, 7]]
    Reasoning: The input is expanded. 0 becomes [0,0,0]. 7 becomes [7,7]. Each row is similarly repeated, expanding the grid.
    Output Grid:
    [[0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 0, 0, 7, 7, 7, 7, 7, 7], [0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 7, 7, 0, 7, 7, 0, 7, 7], [7, 7, 7, 7, 7, 7, 7, 7, 7], [0, 7, 7, 0, 7, 7, 0, 7, 7], [0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 0, 0, 7, 7, 7, 7, 7, 7], [0, 0, 0, 0, 7, 7, 0, 7, 7]]

    Example 2:
    Input Grid:
    [[4, 0, 4], [0, 0, 0], [0, 4, 0]]
    Reasoning: The input grid is expanded following a specific pattern. Each '4' becomes '[4,0,4]', and each '0' becomes '[0,0,0]'. The same expansion logic occurs from columns to rows.
    Output Grid:
    [[4, 0, 4, 0, 0, 0, 4, 0, 4], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 4, 0, 4, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 4, 0, 0, 0, 0]]

    Example 3:
    Input Grid:
    [[0, 9, 9, 1, 9, 9, 9], [0, 0, 9, 1, 9, 9, 0], [9, 0, 9, 1, 9, 9, 0], [0, 0, 0, 1, 9, 0, 0], [0, 9, 9, 1, 9, 9, 9]]
    Reasoning: 1 is kept and is converted to [0, 8, 8]. The rest of the numbers become [0,0,0] or [8,0,0] based on 9s and 0s. The pattern appears to occur such that only 1 numbers are kept from each grid and the rest of the locations in the grid become 0 or 8.
    Output Grid:
    [[0,0,0],[0,0,0],[0,0,0],[0,8,8],[0,0,0]]
    

    Given the training examples and the TEST INPUT below, transform the TEST INPUT according to the patterns observed in the examples. 
    Be sure to output the grid in a list of lists structure.
    {question}
    """

    # Call the LLM
    llm_output = call_llm(prompt)

    # Implement very basic validation: Check for list of lists structure
    if "Error" in llm_output:
        return "Error occurred during processing."
    elif not('[' in llm_output and ']' in llm_output): # rudamentary validation. More can be done!
        return "Invalid format in LLM output."
    else:
        return llm_output

def main(question: str) -> str:
    """
    Main function to solve the grid transformation problem.
    """
    answer = solve_grid_transformation(question)
    return answer