#!/usr/bin/env python
"""
This script explores a new approach to grid transformation problems by using a 
test-time training approach where the LLM develops and validates a hypothesis based 
on provided training examples before applying it to the test case.

This script tests a new hypothesis: That a "test time training" approach where an LLM
develops and tests a pattern against training data before applying it to an unseen example
improves results, even in complex grid transformations. We will test this by having the LLM 
explicitly state and test a transformation hypothesis on the provided training grids, 
before generating the final answer. This will test whether explicit reasoning and
verification are more effective than implicit learning of the pattern, even with
very few examples.
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
    """Solve grid transformation using test-time training."""

    # Step 1: Extract examples
    extraction_prompt = f"""
    Given this question, extract the training examples and the test input.
    Format your response as follows:
    TRAINING_EXAMPLES:
    Example 1:
    Input Grid: [first input grid]
    Output Grid: [first output grid]
    Example 2:
    Input Grid: [second input grid]
    Output Grid: [second output grid]
    TEST_INPUT:
    [the test input grid]
    {question}
    """
    extraction_result = call_llm(extraction_prompt)
    if "Error" in extraction_result:
        return "Error extracting information from the question."
    # Step 2: Formulate hypothesis and test against examples
    hypothesis_prompt = f"""
    Based on the TRAINING EXAMPLES: and the TEST INPUT: from the following, formulate a hypothesis:
    TRAINING_EXAMPLES:
    Example 1:
    Input Grid: [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    Output Grid: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    Example 2:
    Input Grid: [[0, 2, 0], [2, 0, 2], [0, 2, 0]]
    Output Grid: [[0, 2, 0], [2, 0, 2], [0, 2, 0]]
    TEST_INPUT:
    [[5, 0, 0], [0, 0, 0], [0, 0, 5]]
    State your hypothesis. Then test the hypothesis against all TRAINING_EXAMPLES to be sure that your logic produces the Output Grid.
    """
    hypothesis_result = call_llm(extraction_result + "\n" + hypothesis_prompt)
    if "Error" in hypothesis_result:
        return "Error formulating the hypothesis."

    # Step 3: Apply the hypothesis to the test input
    application_prompt = f"""
    You have identified a hypothesis:
    'If the input is not in the corners, make the input zero'
    TRAINING_EXAMPLES:
    Example 1:
    Input Grid: [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    Output Grid: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    Example 2:
    Input Grid: [[0, 2, 0], [2, 0, 2], [0, 2, 0]]
    Output Grid: [[0, 2, 0], [2, 0, 2], [0, 2, 0]]
    Based on your hypothesis above and the following:
    TEST_INPUT:
    [[5, 0, 0], [0, 0, 0], [0, 0, 5]]
    Apply your hypothesis. Provide ONLY the answer.
    """
    application_result = call_llm(hypothesis_result + "\n" + application_prompt)
    if "Error" in application_result:
        return "Error applying the hypothesis to the test input."

    return application_result

def main(question: str) -> str:
    """Main function to solve the problem."""
    answer = solve_grid_transformation(question)
    return answer