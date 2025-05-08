#!/usr/bin/env python
"""
Exploration: Ensemble of Transformation Techniques with Dynamic Weighting

Hypothesis: Combining multiple transformation techniques and dynamically weighting their application based on relevance will improve grid transformation performance.

This approach differs significantly from previous ones by:
1.  Ensembling: Applies multiple transformations, and dynamically combining them to create a final hybrid result.
2.  Dynamic Weighting: Use the LLM to assess and balance the contribution of each transformation technique.
3. Focus on Local vs Global Strategies: This approach uses and weighs both local and global transformation strategies.
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
        extracted_grids = eval(extracted_grids_str)
        return extracted_grids
    except Exception as e:
        print(f"Error parsing extracted grids: {e}")
        return {"train_input": [], "train_output": [], "test_input": []}

def apply_global_transformation(train_input: List, train_output: List, test_input: List) -> str:
    """Applies global transformations such as shifting or rotation to the test input."""
    prompt = f"""You are an expert in global grid transformations.
    Given the training examples (input and output grids) and the test input grid, identify and apply a global transformation (e.g., shifting, rotation, mirroring) to the test input.

    Example:
    train_input: [[[1, 2], [3, 4]]]
    train_output: [[[2, 1], [4, 3]]]
    test_input: [[[5, 6], [7, 8]]]
    Global Transformation: The columns are swapped. Transformed Grid: [[[6, 5], [8, 7]]]

    train_input: {train_input}
    train_output: {train_output}
    test_input: {test_input}
    Global Transformation:
    """
    transformed_grid = call_llm(prompt)
    return transformed_grid

def apply_local_transformation(train_input: List, train_output: List, test_input: List) -> str:
    """Applies local transformations based on neighborhood relationships."""
    prompt = f"""You are an expert in local grid transformations.
    Given the training examples (input and output grids) and the test input grid, identify and apply a local transformation based on neighborhood relationships.

    Example:
    train_input: [[[0, 0], [0, 1]]]
    train_output: [[[1, 1], [1, 1]]]
    test_input: [[[0, 1], [0, 0]]]
    Local Transformation: Non-zero values propagate to all neighbors. Transformed Grid: [[[1, 1], [1, 1]]]

    train_input: {train_input}
    train_output: {train_output}
    test_input: {test_input}
    Local Transformation:
    """
    transformed_grid = call_llm(prompt)
    return transformed_grid

def determine_weights(question: str, global_transformation: str, local_transformation: str) -> str:
    """Determines the weights for combining global and local transformations."""
    prompt = f"""You are an expert in blending grid transformations.
    Given the question and the results of applying global and local transformations, determine the appropriate weights (0 to 1) to combine the results.

    Example:
    question: ... (training examples show a mirroring with local propagation) ...
    global_transformation: Mirroring applied.
    local_transformation: Propagation applied.
    Weights: Global: 0.6, Local: 0.4 (mirroring is more important)

    question: {question}
    global_transformation: {global_transformation}
    local_transformation: {local_transformation}
    Weights:
    """
    weights = call_llm(prompt)
    return weights

def combine_transformations(global_transformation: str, local_transformation: str, weights: str) -> str:
    """Combines the global and local transformations based on the determined weights."""
    prompt = f"""You are an expert at blending grid transformations.
    Combine the global and local transformations based on the given weights to produce the final transformed grid.

    Example:
    global_transformation: [[[6, 5], [8, 7]]]
    local_transformation: [[[1, 1], [1, 1]]]
    weights: Global: 0.6, Local: 0.4
    Combined Transformation: [[[4, 3], [5, 5]]]

    global_transformation: {global_transformation}
    local_transformation: {local_transformation}
    weights: {weights}
    Combined Transformation:
    """
    combined_grid = call_llm(prompt)
    return combined_grid

def main(question: str) -> str:
    """Main function to solve the problem."""
    try:
        # 1. Extract grids
        extracted_grids = extract_grids(question)
        if not extracted_grids["train_input"] or not extracted_grids["train_output"] or not extracted_grids["test_input"]:
            return "Error: Could not extract all necessary grids."

        train_input = extracted_grids["train_input"]
        train_output = extracted_grids["train_output"]
        test_input = extracted_grids["test_input"]

        # 2. Apply global transformation
        global_transformation = apply_global_transformation(train_input, train_output, test_input)

        # 3. Apply local transformation
        local_transformation = apply_local_transformation(train_input, train_output, test_input)

        # 4. Determine weights
        weights = determine_weights(question, global_transformation, local_transformation)

        # 5. Combine transformations
        combined_grid = combine_transformations(global_transformation, local_transformation, weights)
        return combined_grid
    except Exception as e:
        return f"An error occurred: {e}"