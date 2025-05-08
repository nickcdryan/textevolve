#!/usr/bin/env python
"""
This script introduces a new approach to solving grid transformation problems, focusing on
context-based reasoning and adaptive example selection. The problem is framed as a contextual
understanding and transformation task, where the LLM identifies the context, selects
relevant examples, and applies transformations accordingly.

Hypothesis: Context-based reasoning with adaptive example selection allows the LLM to better
understand the underlying transformation logic and generalize to new inputs. Adaptive example
selection helps mitigate the impact of noisy or irrelevant examples by focusing on the most
relevant ones for a given context.
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

def identify_context(question: str) -> str:
    """Identify the context of the grid transformation problem."""
    prompt = f"""
    You are an expert grid context identifier.
    Identify the key context of this grid transformation question.
    Example:
    Question: === TRAINING EXAMPLES === ... The context is expansion of number in the location to the sides.
    Question: {question}
    Context:
    """
    return call_llm(prompt)

def select_relevant_examples(question: str, context: str) -> List[str]:
    """Select relevant examples based on the identified context."""
    prompt = f"""
    You are an expert example selector.
    Given the question and context, select the 3 most relevant examples from the training examples.

    Question: {question}
    Context: {context}

    Example:
    Question: ... Transformation involves shifting right ... Example 1 is the only valid example to apply.

    Relevant Examples: (List the numbers of the examples to apply)

    """
    return call_llm(prompt)

def apply_transformation(question: str, relevant_examples: List[str]) -> str:
    """Apply the transformation to the test input based on the selected examples."""
    prompt = f"""
    You are an expert grid transformer.
    Given the question and relevant examples, apply the transformation to the test input.

    Question: {question}
    Relevant Examples: {relevant_examples}
    Here is how it should perform, using the same question format:
    Example:
        Question:
            === TRAINING EXAMPLES ===
            Example 1:
                Input Grid: [[1, 2], [3, 4]]
                Output Grid: [[2, 3], [4, 1]]
            === TEST INPUT ===
            [[5, 6], [7, 8]]
            Transform the test input according to the pattern shown in the training examples.

    New Grid:
    [[6, 7], [8, 5]]

    Please apply the rule and return the NEW Extracted Grid.
    """
    new_grid = call_llm(prompt)
    return new_grid

def verify_grid(question: str, new_grid: str) -> str:
    """Verify the transformation logic and transformation against the original questions to look for errors."""
    prompt = f"""
    You are an expert grid verifier.  You must verify that a transformation is valid, by performing error analysis.

    question: {question}
    transformation: {new_grid}

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

    Example of a incorrect transformation, with explanation.
        question:
            === TRAINING EXAMPLES ===
            Example 1:
                Input Grid: [[1, 2], [3, 4]]
                Output Grid: [[2, 3], [4, 1]]
            === TEST INPUT ===
            [[5, 6], [7, 8]]
            Transform the test input according to the pattern shown in the training examples.

    transformation: [[6, 7], [8, 6]]
    verified: INCORRECT because all numbers must shift, 6 must become 5

    Please verify the new grid and say if it is correct.
    """
    verified = call_llm(prompt)
    return verified

def main(question: str) -> str:
    """Main function to solve the problem."""
    try:
        # 1. Identify the context
        context = identify_context(question)

        # 2. Select relevant examples
        relevant_examples_text = select_relevant_examples(question, context)
        relevant_examples = re.findall(r'\d+', relevant_examples_text)

        # 3. Apply the transformation
        transformed_grid = apply_transformation(question, relevant_examples)

        # 4. Verify transformation and new grid
        verified = verify_grid(question, transformed_grid)

        if "INCORRECT" in verified:
            return f"Error: Transformation verification failed. {verified}"

        return transformed_grid
    except Exception as e:
        return f"An error occurred: {e}"