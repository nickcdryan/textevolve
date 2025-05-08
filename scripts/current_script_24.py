#!/usr/bin/env python
"""
Exploration: Analogy-Based Grid Transformation with Dynamic Example Selection.

Hypothesis: Leveraging an analogy-based approach with dynamic example selection will improve grid transformation performance.
This approach will identify relevant training examples based on similarity to the test input and use them to guide the transformation process.

This approach differs significantly from previous ones by:
1. Using Analogy-Based Reasoning: Transform the grid not just by pattern matching, but by analogy to existing examples.
2. Dynamic Example Selection: Select the most relevant training examples to use as "analogies" for the given test input.
3. Focus on Structural Similarity: Prioritize structural characteristics of the grid (size, density, etc.) to guide example selection.
4. Apply a similarity weighting between the top analogies, and apply the weighted transformation
"""

import os
import re
from typing import List, Dict, Any, Optional, Union

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response. This is how you call the LLM."""
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

def select_relevant_examples(question: str, num_examples: int = 2) -> str:
    """Selects the most relevant training examples based on similarity to the test input."""
    prompt = f"""
    You are an expert at selecting relevant examples. Given the following question, select the {num_examples} most relevant training examples based on their structural similarity to the test input grid. Structural similarity includes grid size, density of non-zero elements, and general arrangement of elements.

    Example:
    question:
    === TRAINING EXAMPLES ===
    Example 1:
    Input Grid: [[0, 0], [0, 1]]
    Output Grid: [[1, 1], [1, 1]]
    Example 2:
    Input Grid: [[1, 1], [1, 0]]
    Output Grid: [[0, 0], [0, 0]]
    === TEST INPUT ===
    [[0, 1], [0, 0]]
    Transform the test input.

    Relevant Examples: Examples 1 and 2 (both are 2x2 grids with a mix of 0 and 1 values).

    question: {question}
    Relevant Examples:
    """
    relevant_examples = call_llm(prompt)
    return relevant_examples

def analogy_based_transformation(question: str, relevant_examples: str) -> str:
    """Transforms the test input based on analogies drawn from the relevant examples."""
    prompt = f"""
    You are an expert grid transformation agent. Given the question and the relevant examples, transform the test input based on analogies drawn from the examples. Weigh the transformations on the most relevant analogies.

    Example:
    question:
    === TRAINING EXAMPLES ===
    Example 1:
    Input Grid: [[0, 0], [0, 1]]
    Output Grid: [[1, 1], [1, 1]]
    Example 2:
    Input Grid: [[1, 1], [1, 0]]
    Output Grid: [[0, 0], [0, 0]]
    === TEST INPUT ===
    [[0, 1], [0, 0]]
    Transform the test input.

    Relevant Examples: Examples 1 and 2

    Transformed Grid: [[1, 0], [1, 0]] (Applying rule from Example 1 to row 0, Example 2 to row 1)

    question: {question}
    Relevant Examples: {relevant_examples}
    Transformed Grid:
    """
    transformed_grid = call_llm(prompt)
    return transformed_grid

def verify_transformation(question: str, transformed_grid: str) -> str:
    """Verify transformation is correct given training examples."""
    prompt = f"""You are a verification agent, making sure transformations are correct.
    Verify if the transformation is correct based on the training examples in the question.

    Example:
    question: === TRAINING EXAMPLES === Example 1: Input Grid: [[1, 2], [3, 4]] Output Grid: [[2, 1], [4, 3]] === TEST INPUT === [[5, 6], [7, 8]] Transform the test input.
    transformed_grid: [[6, 5], [8, 7]]
    Is the transformation correct? Yes, the columns were swapped.

    question: {question}
    transformed_grid: {transformed_grid}
    Is the transformation correct?"""
    is_correct = call_llm(prompt)
    return is_correct

def main(question: str) -> str:
    """Main function to solve the problem."""
    try:
        # 1. Select relevant examples
        relevant_examples = select_relevant_examples(question)

        # 2. Apply analogy-based transformation
        transformed_grid = analogy_based_transformation(question, relevant_examples)

        # 3. Verify the transformation
        is_correct = verify_transformation(question, transformed_grid)

        if "No" in is_correct:
            return "Transformation incorrect. Check the training examples"
        else:
            return transformed_grid
    except Exception as e:
        return f"An error occurred: {e}"