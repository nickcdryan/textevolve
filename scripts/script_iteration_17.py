#!/usr/bin/env python
"""This script solves grid transformation problems using a new LLM-driven approach that combines pattern identification, contextual analysis, and targeted transformation. The core idea is to identify key elements in the input grid and apply transformations based on their context, leveraging test-time analysis of training examples to guide the transformation.

This approach differs from previous ones by:

1.  Focusing on test-time analysis, where the training examples are re-analyzed during the processing of the test input to identify the most relevant transformation rule.
2.  Using targeted element transformation, where the identified key elements (e.g., numbers) and their context (e.g., neighbors, position) guide the transformation process.

It relies heavily on multiple examples and targeted prompt engineering to guide the LLM in performing the transformation."""

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

def analyze_training_examples(question: str) -> str:
    """Analyze training examples to identify key transformation elements and patterns."""
    prompt = f"""You are an expert at analyzing grid transformation problems.
    Analyze the training examples in the provided question to identify:
    1.  Key elements that are transformed (e.g., specific numbers, patterns).
    2.  The context in which these elements are transformed (e.g., neighbors, position).
    3.  The general transformation rule that applies to these elements.

    Example:
    question: === TRAINING EXAMPLES === Example 1: Input Grid: [[1, 2], [3, 4]] Output Grid: [[4, 3], [2, 1]] === TEST INPUT === [[5, 6], [7, 8]] Transform the test input.
    Analysis:
    Key elements: All numbers in the grid.
    Context: Their position in the 2x2 grid.
    Transformation rule: Flip the grid horizontally and vertically.

    question: {question}
    Analysis:"""
    analysis = call_llm(prompt)
    return analysis

def apply_transformation(input_grid: str, analysis: str) -> str:
    """Apply the identified transformation rule to the input grid, focusing on identified elements and their context."""
    prompt = f"""You are an expert grid transformation agent.
    Apply the transformation rule described in the analysis to the provided input grid.
    Focus on the key elements and their context to guide the transformation.
    Consider the surrounding data or grid formatting as well.
    The goal is to generate a grid that properly represents the transformation rule.

    Example:
    input_grid: [[5, 6], [7, 8]]
    analysis:
    Key elements: All numbers in the grid.
    Context: Their position in the 2x2 grid.
    Transformation rule: Flip the grid horizontally and vertically.
    Transformed grid: [[8, 7], [6, 5]]

    input_grid: {input_grid}
    analysis: {analysis}
    Transformed grid:"""
    transformed_grid = call_llm(prompt)
    return transformed_grid

def verify_transformation(question: str, transformed_grid: str) -> str:
  """Verify that the transformation is valid by performing error analysis."""
  prompt = f"""You are an expert grid transformation verifier. Verify that the new grid provided makes sense given the question.
  Here is how it should perform, using the same question format:
  Example of a valid transformation, with explanation.
        question:
            === TRAINING EXAMPLES ===
            Example 1:
                Input Grid: [[1, 2], [3, 4]]
                Output Grid: [[2, 3], [4, 1]]
            === TEST INPUT ===
            [[5, 6], [7, 8]]
            Transform the test input according to the pattern shown in the training examples.

    transformation: [[6, 7], [8, 5]]
    verified: CORRECT because numbers shift to the right.

    question: {question}
    transformation: {transformed_grid}
    verified: 
    """
  verified = call_llm(prompt)
  return verified

def main(question: str) -> str:
    """Main function to solve the problem."""
    try:
        # 1. Analyze the training examples to identify key elements and transformation rules.
        analysis = analyze_training_examples(question)

        # 2. Extract the test input grid.
        test_input_match = re.search(r"=== TEST INPUT ===\n(.*?)\nTransform", question, re.DOTALL)
        if not test_input_match:
            return "Error: Could not find TEST INPUT in the question."
        input_grid = test_input_match.group(1).strip()

        # 3. Apply the transformation rule to the test input grid, focusing on identified elements and their context.
        transformed_grid = apply_transformation(input_grid, analysis)

        # 4. Verify the transformation
        verified = verify_transformation(question, transformed_grid)

        if "INCORRECT" in verified:
            return f"Error: Transformation verification failed. {verified}"

        return transformed_grid
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred: {e}"