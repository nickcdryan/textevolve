import os
import re
import math

# Hypothesis: This exploration will implement a "Constraint-Based Transformation with Iterative Region Analysis" approach.
# The LLM will identify "stable regions" (unchanged in training examples) and "transformation regions" (changed).
# It will then generate constraints for the transformation based on the stable regions, and apply transformations only to the unstable regions.
# Also, add verification steps to understand which parts are successful and where it is breaking.

def main(question):
    """Transforms a grid based on constraint-based transformation and region analysis."""
    try:
        # 1. Extract training examples and test input
        training_examples, test_input = preprocess_question(question)

        # 2. Identify stable and transformation regions
        stable_regions, transformation_regions = analyze_regions(training_examples)

        # 3. Generate transformation constraints based on stable regions
        transformation_constraints = generate_constraints(stable_regions)

        # 4. Apply transformations to transformation regions based on constraints
        transformed_grid = apply_transformations(test_input, transformation_regions, transformation_constraints)

        return transformed_grid

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def preprocess_question(question):
    """Extract training examples and test input from the question string using regex."""
    try:
        training_examples_match = re.search(r"=== TRAINING EXAMPLES ===\n(.*?)\n=== TEST INPUT ===", question, re.DOTALL)
        test_input_match = re.search(r"=== TEST INPUT ===\n(.*?)\nTransform", question, re.DOTALL)

        training_examples = training_examples_match.group(1).strip() if training_examples_match else ""
        test_input = test_input_match.group(1).strip() if test_input_match else ""

        return training_examples, test_input
    except Exception as e:
        return "", ""

def analyze_regions(training_examples):
    """Identifies stable and transformation regions in the training examples using LLM."""
    system_instruction = "You are an expert in analyzing grid transformations to identify stable and transformation regions."
    prompt = f"""
    You are an expert in analyzing grid transformations to identify stable and transformation regions. Stable regions are areas of the grid that remain unchanged across all training examples, and transformation regions are areas that change.

    Example:
    Training Examples:
    Input Grid: [[0, 1], [0, 0]]
    Output Grid: [[0, 1], [1, 1]]
    Stable Regions: [[0, 1]] (top row)
    Transformation Regions: [[0, 0] -> [1, 1]] (bottom row)

    Now, for this new task:
    Training Examples:
    {training_examples}
    Stable Regions and Transformation Regions:
    """
    regions_analysis = call_llm(prompt, system_instruction)
    # Assuming the LLM provides structured output for stable and transformation regions.
    return regions_analysis, regions_analysis  # Placeholder - needs actual parsing

def generate_constraints(stable_regions):
    """Generates transformation constraints based on the stable regions using LLM."""
    system_instruction = "You are an expert in generating transformation constraints based on stable regions in grid data."
    prompt = f"""
    You are an expert in generating transformation constraints based on stable regions in grid data. Transformation constraints are rules that must be followed when transforming the grid, based on the stable regions.

    Example:
    Stable Regions: [[0, 1]] (top row)
    Transformation Constraints: The top row must remain unchanged during the transformation.

    Now, for these stable regions:
    {stable_regions}
    Transformation Constraints:
    """
    transformation_constraints = call_llm(prompt, system_instruction)
    return transformation_constraints

def apply_transformations(test_input, transformation_regions, transformation_constraints):
    """Applies transformations to the transformation regions based on the constraints using LLM."""
    system_instruction = "You are an expert in applying grid transformations based on transformation regions and constraints."
    prompt = f"""
    You are an expert in applying grid transformations based on transformation regions and constraints. Apply transformations to the transformation regions based on the transformation constraints, ensuring that the constraints are followed.

    Example:
    Test Input: [[0, 0], [0, 0]]
    Transformation Regions: [[0, 0]] (bottom row)
    Transformation Constraints: The top row must remain unchanged.
    Transformed Grid: [[0, 0], [1, 1]]

    Now, for this new task:
    Test Input:
    {test_input}
    Transformation Regions:
    {transformation_regions}
    Transformation Constraints:
    {transformation_constraints}
    Transformed Grid:
    """
    transformed_grid = call_llm(prompt, system_instruction)
    return transformed_grid

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response. DO NOT deviate from this example template or invent configuration options. This is how you call the LLM."""
    try:
        from google import genai
        from google.genai import types
        import os

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