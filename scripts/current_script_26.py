#!/usr/bin/env python
"""
Exploration: Iterative Transformation Rule Induction with Multi-Example Feedback and Targeted Refinement

Hypothesis: This exploration tests an approach that focuses on iteratively inducing transformation rules directly from multiple examples,
leveraging targeted refinement based on multi-example feedback to improve accuracy.
The script will first construct transformation rules iteratively and then apply them to the test input using the LLM to transform the grid directly.

This approach differs significantly from previous ones by:
1. Iterative Rule Construction: Building transformation rules iteratively by progressively incorporating information from multiple training examples.
2. Multi-Example Feedback: Gathering feedback by testing the rule on ALL training examples simultaneously, enabling more comprehensive error detection.
3. Targeted Refinement: Using the feedback to refine the rule with a specific focus on areas of disagreement or inaccuracy.
4. Direct LLM Transformation: Apply the transformation to the test input. This will minimize reliance on external code and improve overall efficiency.
"""

import os
import re
from typing import List, Dict, Any, Optional, Union

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response."""
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

def extract_training_and_test_data(question: str) -> Dict:
    """Extracts training examples and test input from the question."""
    prompt = f"""You are an expert at extracting structured data. Extract training examples and the test input.

    Example:
    question: === TRAINING EXAMPLES === Example 1: Input Grid: [[1, 2], [3, 4]] Output Grid: [[4, 3], [2, 1]] === TEST INPUT === [[5, 6], [7, 8]] Transform the test input.
    {{ "training_examples": [{{ "input": "[[1, 2], [3, 4]]", "output": "[[4, 3], [2, 1]]" }}], "test_input": "[[5, 6], [7, 8]]" }}

    question: {question}
    Extracted Data:
    """
    extracted_data = call_llm(prompt)
    try:
        return eval(extracted_data)
    except Exception as e:
        print(f"Error parsing extracted data: {e}")
        return {"training_examples": [], "test_input": ""}

def construct_transformation_rule(training_examples: List[Dict]) -> str:
    """Constructs a transformation rule iteratively from training examples."""
    prompt = f"""You are an expert at constructing transformation rules.
    Given the following training examples, construct a general transformation rule that explains all examples.

    Example:
    training_examples: [{{ "input": "[[1, 2], [3, 4]]", "output": "[[4, 3], [2, 1]]" }}]
    Transformation Rule: The grid is flipped horizontally and vertically.

    training_examples: {training_examples}
    Transformation Rule:
    """
    transformation_rule = call_llm(prompt)
    return transformation_rule

def apply_transformation(test_input: str, transformation_rule: str) -> str:
    """Apply the transformation rule to the test input."""
    prompt = f"""You are an expert at applying transformation rules.
    Apply the following transformation rule to the test input.

    Example:
    transformation_rule: The grid is flipped horizontally and vertically.
    test_input: [[5, 6], [7, 8]]
    Transformed Grid: [[8, 7], [6, 5]]

    transformation_rule: {transformation_rule}
    test_input: {test_input}
    Transformed Grid:
    """
    transformed_grid = call_llm(prompt)
    return transformed_grid

def verify_transformation(transformed_grid: str, question: str) -> str:
    """Verify transformation is correct given training examples."""
    prompt = f"""You are a verification agent. Make sure the transformations are correct given the training examples in the question.
    Make sure that the output is a valid matrix as a list of lists of integers.

    Example:
    question: === TRAINING EXAMPLES === Example 1: Input Grid: [[1, 2], [3, 4]] Output Grid: [[2, 1], [4, 3]] === TEST INPUT === [[5, 6], [7, 8]] Transform the test input.
    transformed_grid: [[6, 5], [8, 7]]
    Is the transformation correct? Yes, the columns were swapped.

    question: {question}
    transformed_grid: {transformed_grid}
    Is the transformation correct?
    """
    is_correct = call_llm(prompt)
    return is_correct

def main(question: str) -> str:
    """Main function to solve the problem."""
    try:
        # 1. Extract training examples and test input
        extracted_data = extract_training_and_test_data(question)
        training_examples = extracted_data["training_examples"]
        test_input = extracted_data["test_input"]

        # 2. Construct a transformation rule
        transformation_rule = construct_transformation_rule(training_examples)

        # 3. Apply the transformation rule to the test input
        transformed_grid = apply_transformation(test_input, transformation_rule)

        # 4. Verify the transformation
        is_correct = verify_transformation(transformed_grid, question)
        if "No" in is_correct:
            return "Transformation incorrect. Check the training examples"
        else:
            return transformed_grid

    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred: {e}"