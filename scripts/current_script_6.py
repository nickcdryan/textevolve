#!/usr/bin/env python
"""
This script introduces a novel approach to solving grid transformation problems
by focusing on identifying and applying local structural motifs. It is inspired by
image processing techniques that look for recurring patterns to transform and clean up images.

Hypothesis: Identifying repeating sub-structures ("motifs") in the input grid and mapping their
transformation to the output grid can provide a robust way to generalize transformations, even
with limited examples. This approach seeks to go beyond simple pattern matching by understanding
the relationship between these motifs and their transformations.

This script attempts a new approach:
1. Motifs are recognized as repeating subgrids.
2. The relationship between these motifs across training examples are analyzed to
deduce transformations.
3. A 'transformation' in this sense means the relationship between a subgrid and its new version in the ouptut grids.
"""

import os
import re
from typing import List, Dict, Any, Optional, Union

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response. """
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

def extract_motifs_and_transformations(question: str) -> str:
    """Extract repeating motifs and their transformations from the training examples."""
    prompt = f"""
    You are an expert grid analyst. Your task is to identify repeating sub-structures ("motifs") within the training examples
    and deduce how these motifs are transformed from the input grid to the output grid. Focus on recurring arrangements of
    numbers.

    Example:
    Question:
    === TRAINING EXAMPLES ===
    Example 1:
    Input Grid: [[1, 2, 1], [2, 1, 2], [1, 2, 1]]
    Output Grid: [[2, 1, 2], [1, 2, 1], [2, 1, 2]]
    Example 2:
    Input Grid: [[3, 4, 3], [4, 3, 4], [3, 4, 3]]
    Output Grid: [[4, 3, 4], [3, 4, 3], [4, 3, 4]]
    === TEST INPUT ===
    [[5, 6, 5], [6, 5, 6], [5, 6, 5]]

    Analysis:
    Motif: The alternating pattern [[A, B, A], [B, A, B], [A, B, A]] where A and B are distinct numbers.
    Transformation: Swap the positions of A and B within the motif.

    Question:
    {question}

    Identify repeating motifs and describe how they are transformed. Be concise and specific in your analysis.

    """
    analysis = call_llm(prompt)
    return analysis

def apply_motif_transformation(input_grid: str, motif_analysis: str) -> str:
    """Apply the identified motif transformations to the test input grid."""
    prompt = f"""
    You are a skilled grid transformer. Given the input grid and the analysis of motifs and their transformations,
    apply the transformations to generate the output grid.

    Input Grid:
    {input_grid}

    Motif Analysis:
    {motif_analysis}

    Example:
    Input Grid:
    [[5, 6, 5], [6, 5, 6], [5, 6, 5]]
    Motif Analysis:
    Motif: The alternating pattern [[A, B, A], [B, A, B], [A, B, A]] where A and B are distinct numbers.
    Transformation: Swap the positions of A and B within the motif.
    Output Grid:
    [[6, 5, 6], [5, 6, 5], [6, 5, 6]]

    Based on the motif analysis, generate the transformed grid. Ensure the output grid is correctly formatted. Provide ONLY the grid.
    """
    output_grid = call_llm(prompt)
    return output_grid

def verify_output_format(output_grid: str) -> str:
  """Verify the format of the output grid."""
  prompt = f"""
  You are an expert grid format verifier. Determine if the following output_grid is correctly formatted as a 2D list of integers.

  Example of a correct grid:
  output_grid: [[1, 2], [3, 4]]
  verified: CORRECT

  Here are examples of incorrect grids:
  output_grid: [1, 2], [3, 4]
  verified: INCORRECT

  output_grid: "[[1, 2], [3, 4]]"
  verified: INCORRECT

  output_grid: [[1, 2], [3, 4]
  verified: INCORRECT

  Here's the input:
  output_grid: {output_grid}
  verified:
  """
  verified = call_llm(prompt)
  return verified

def main(question: str) -> str:
    """Main function to solve the problem."""
    try:
        # 1. Extract the test input grid from the question
        test_input_match = re.search(r"=== TEST INPUT ===\n(.*?)\nTransform", question, re.DOTALL)
        if not test_input_match:
            return "Error: Could not find TEST INPUT in the question."
        input_grid = test_input_match.group(1).strip()

        # 2. Extract motifs and transformations
        motif_analysis = extract_motifs_and_transformations(question)

        # 3. Apply the transformations to the test input grid
        output_grid = apply_motif_transformation(input_grid, motif_analysis)

        # 4. Verify the format of the output grid
        verified = verify_output_format(output_grid)

        if "INCORRECT" in verified:
          return f"Error: Output grid format is incorrect. {output_grid}"

        return output_grid
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An unexpected error occurred."