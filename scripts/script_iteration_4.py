import os
import re
import math
import json

# New approach: Visual analogy reasoning with structured rule representation
# Hypothesis: Representing transformation rules in a more structured way (key-value pairs)
# and using visual analogy reasoning will improve performance

def main(question):
    """
    Transform a grid based on visual analogy reasoning with structured rule representation.
    """
    try:
        # Decompose the question into training examples and test input
        training_examples, test_input = split_question(question)

        # Identify transformation rule by visual comparison of training examples and represent as key-value pairs
        transformation_rule = identify_transformation_rule(training_examples)

        # Apply the transformation rule to the test input
        transformed_grid = apply_transformation(test_input, transformation_rule)

        return transformed_grid

    except Exception as e:
        return f"An error occurred: {str(e)}"

def split_question(question):
    """Splits the question into training examples and test input."""
    try:
        training_examples_str = question.split("=== TEST INPUT ===")[0]
        test_input_str = question.split("=== TEST INPUT ===")[1]
        return training_examples_str, test_input_str
    except IndexError as e:
        return "Error: Missing separator", ""

def identify_transformation_rule(training_examples):
    """Identify the transformation rule from training examples by visual comparison and represent it as structured rules."""
    prompt = f"""
    You are an expert visual pattern recognition system. Analyze the training examples and identify the transformation rule. Represent the rule as a set of key-value pairs where the key is a pattern in the input grid, and the value is its corresponding transformation in the output grid.

    Example:
    Input Grid:
    [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
    Output Grid:
    [[1, 1, 1], [0, 0, 0], [1, 1, 1]]
    Structured Transformation Rule:
    {{
      "Invert Grid": "Change 0 to 1 and 1 to 0"
    }}
    
    Training Examples:
    {training_examples}
    
    Identify the transformation rule by visually comparing the input and output grids. Represent the most significant changes as structured key-value pairs.
    Structured Transformation Rule:
    """
    
    # Call the LLM
    transformation_rule = call_llm(prompt, system_instruction="You are a visual pattern recognition expert.")
    return transformation_rule

def apply_transformation(test_input, transformation_rule):
    """Apply the transformation rule to the test input using visual analogy."""
    prompt = f"""
    Apply the transformation rule to the test input grid using visual analogy.

    Example:
    Test Input:
    [[0, 1], [1, 0]]
    Structured Transformation Rule:
    {{
      "Invert Grid": "Change 0 to 1 and 1 to 0"
    }}
    Transformed Grid:
    [[1, 0], [0, 1]]
    
    Test Input:
    {test_input}
    Structured Transformation Rule:
    {transformation_rule}

    Apply the transformation rule to the test grid using visual analogy and generate the transformed grid. Use plain text for the resulting grid.
    Transformed Grid:
    """
    transformed_grid = call_llm(prompt, system_instruction="You are an expert grid transformer using visual analogy.")

    # Verification: check if the output is in expected format
    if not transformed_grid:
        return "Error: No transformation occurred"
    
    # Basic format verification
    if not ("[[" in transformed_grid and "]]" in transformed_grid):
      return "Error: output grid is not in standard format"

    return transformed_grid