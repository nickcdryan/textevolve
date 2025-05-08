#!/usr/bin/env python
"""
Exploration: Decomposition with Targeted Sub-Problems.
Hypothesis: Decomposing grid transformation problems into specific sub-problems (feature identification, structural transformation, value mapping)
and assigning focused LLM calls to each will improve accuracy. By using validation checks at each stage, we can identify breaking points and fix them.

This approach differs significantly from previous ones by:
1. Decomposing the problem into three core, focused sub-problems with specific prompts for each.
2. Performing an explicit extraction of training grid inputs, outputs, and the testing grid inputs. This reduces the chance of parsing errors.
3. Explicit validation steps at each major phase of processing.
4. Directly transforming the test input to reduce the need for complex code.
"""

import os
import re
from typing import List, Dict, Any, Optional, Union

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt."""
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

def extract_grids(question: str) -> Dict:
    """Extracts the training and test grids from the question."""
    prompt = f"""
    You are an expert at extracting information from grid transformation problems.
    Given the following question, extract all training input grids, training output grids, and the test input grid.

    Example:
    question: === TRAINING EXAMPLES === Example 1: Input Grid: [[1, 2], [3, 4]] Output Grid: [[4, 3], [2, 1]] === TEST INPUT === [[5, 6], [7, 8]] Transform the test input.
    Extracted Grids: {{"train_input": [[[1, 2], [3, 4]]], "train_output": [[[4, 3], [2, 1]]], "test_input": [[[5, 6], [7, 8]]]}}

	question: === TRAINING EXAMPLES === Example 1: Input Grid: [[0, 0], [0, 4]] Output Grid: [[4, 4], [4, 4]] === TEST INPUT === [[0, 0], [0, 0]] Transform the test input.
    Extracted Grids: {{"train_input": [[[0, 0], [0, 4]]], "train_output": [[[4, 4], [4, 4]]], "test_input": [[[0, 0], [0, 0]]]}}

    question: {question}
    Extracted Grids:
    """
    extracted_grids_str = call_llm(prompt)
    try:
        extracted_grids = eval(extracted_grids_str) # Convert extracted grids from string format to python object
        return extracted_grids
    except Exception as e:
        print(f"Error parsing extracted grids: {e}")
        return {"train_input": [], "train_output": [], "test_input": []} # Return empty lists to prevent errors

def identify_features(train_input: List, train_output: List) -> str:
    """Identifies key features and transformation logic from the training examples."""
    prompt = f"""
    You are an expert at identifying patterns and features in grid transformations.
    Given the following training input and output grids, identify key features and transformation logic.

    Example:
    train_input: [[[1, 2], [3, 4]]]
    train_output: [[[4, 3], [2, 1]]]
    Identified Features and Logic: The grid is flipped horizontally and vertically.

    train_input: {train_input}
    train_output: {train_output}
    Identified Features and Logic:
    """
    identified_features = call_llm(prompt)
    return identified_features

def structural_transformation(test_input: List, identified_features: str) -> str:
    """Applies structural transformations to the test input based on identified features."""
    prompt = f"""
    You are an expert at applying structural transformations to grids.
    Given the following test input and identified features, apply the necessary structural transformations.

    Example:
    test_input: [[[5, 6], [7, 8]]]
    identified_features: The grid is flipped horizontally and vertically.
    Transformed Grid: [[[8, 7], [6, 5]]]

    test_input: {test_input}
    identified_features: {identified_features}
    Transformed Grid:
    """
    transformed_grid = call_llm(prompt)
    return transformed_grid

def value_mapping(transformed_grid: str, train_input: List, train_output: List) -> str:
    """Maps values in the transformed grid based on training example relationships."""
    prompt = f"""
    You are an expert at mapping values in grid transformations.
    Given the following transformed grid, training input, and training output, map the values in the transformed grid based on the relationships learned from the training examples.

    Example:
    transformed_grid: [[[8, 7], [6, 5]]]
    train_input: [[[1, 2], [3, 4]]]
    train_output: [[[4, 3], [2, 1]]]
    Value Mappings: Based on the training examples, the value 5 maps to 1, 6 maps to 2, 7 maps to 3, and 8 maps to 4.

    transformed_grid: {transformed_grid}
    train_input: {train_input}
    train_output: {train_output}
    Value Mappings:
    """
    value_mappings = call_llm(prompt)
    return value_mappings

def main(question: str) -> str:
    """Main function to solve the problem."""
    try:
        # 1. Extract grids
        extracted_grids = extract_grids(question)
        if not extracted_grids["train_input"] or not extracted_grids["train_output"] or not extracted_grids["test_input"]:
            return "Error: Could not extract all necessary grids."

        # 2. Identify features
        identified_features = identify_features(extracted_grids["train_input"], extracted_grids["train_output"])
        if "Error" in identified_features:
            return f"Error: Could not identify features. {identified_features}"

        # 3. Apply structural transformation
        transformed_grid = structural_transformation(extracted_grids["test_input"], identified_features)
        if "Error" in transformed_grid:
            return f"Error: Could not apply structural transformation. {transformed_grid}"

        # 4. Map values
        value_mappings = value_mapping(transformed_grid, extracted_grids["train_input"], extracted_grids["train_output"])
        if "Error" in value_mappings:
            return f"Error: Could not apply value mappings. {value_mappings}"

        return value_mappings # Return value mapping since its the end of the pipe

    except Exception as e:
        return f"An error occurred: {e}"