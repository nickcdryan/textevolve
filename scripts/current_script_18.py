import os
import re
import math

# This script takes a fundamentally different approach: decomposing the grid transformation into feature extraction,
# transformation selection, and transformation application steps.
# The hypothesis is that by explicitly extracting features, selecting a transformation type,
# and then applying it, we can improve accuracy. It incorporates a multi-example approach and enhanced validation.

def main(question):
    """Transforms a grid based on features and selected transformation."""
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem by extracting features, selecting transformation, and applying it."""

    system_instruction = "You are an expert at grid transformation, able to extract features, select a transformation, and apply it to new grids."

    # STEP 1: Extract features from the problem text
    feature_extraction_prompt = f"""
    You are tasked with extracting key features from the grid transformation problem description.

    Example 1:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[\n  [1, 2]\n  [3, 8]\n]\n\nOutput Grid:\n[\n  [0, 1, 2, 0]\n  [1, 1, 2, 2]\n  [3, 3, 8, 8]\n  [0, 3, 8, 0]\n]\n\n=== TEST INPUT ===\n[\n  [2, 8]\n  [1, 4]\n]\n\n
    Extracted Features: input grid size:2x2, expansion pattern, addition of zero border

    Example 2:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[\n  [4, 4]\n  [4, 4]\n]\n\nOutput Grid:\n[\n  [4]\n]\n\n=== TEST INPUT ===\n[\n  [4, 4, 9, 9]\n  [4, 4, 4, 4]\n]\n\n
    Extracted Features: compression, removal of duplicate values

    Problem: {problem_text}
    Extracted Features:
    """

    extracted_features = call_llm(feature_extraction_prompt, system_instruction)
    print(f"Extracted features: {extracted_features}")

    # STEP 2: Transformation selection based on the extracted features
    transformation_selection_prompt = f"""
    You are given features extracted from a grid transformation problem, and your task is to select the most appropriate transformation.
    
    Features: {extracted_features}

    Example 1:
    Features: input grid size:2x2, expansion pattern, addition of zero border
    Transformation Selected: Expansion with border

    Example 2:
    Features: compression, removal of duplicate values
    Transformation Selected: Compression removing duplicates

    Select the transformation that is most appropriate for the problem.
    Transformation Selected:
    """

    transformation_selected = call_llm(transformation_selection_prompt, system_instruction)
    print(f"Transformation Selected: {transformation_selected}")

    # STEP 3: Transformation Application
    transformation_application_prompt = f"""
    You are an expert in applying grid transformations. Given the original grid, extracted features, and selected transformation, generate the transformed grid.

    Problem: {problem_text}
    Features: {extracted_features}
    Transformation Selected: {transformation_selected}

    Example:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[\n  [1, 2]\n  [3, 8]\n]\n\nOutput Grid:\n[\n  [0, 1, 2, 0]\n  [1, 1, 2, 2]\n  [3, 3, 8, 8]\n  [0, 3, 8, 0]\n]\n\n=== TEST INPUT ===\n[\n  [2, 8]\n  [1, 4]\n]\n\n
    Features: input grid size:2x2, expansion pattern, addition of zero border
    Transformation Selected: Expansion with border
    Transformed Grid: [[0,2,8,0],[2,2,8,8],[1,1,4,4],[0,1,4,0]]

    Transformed Grid:
    """

    transformed_grid_text = call_llm(transformation_application_prompt, system_instruction)
    print(f"Transformed Grid: {transformed_grid_text}")

    # STEP 4: Validation - Verifies if transformed grid is well-formed and reasonable.
    validation_prompt = f"""
    Validate whether the transformed grid is well-formed and reasonable for the original grid and transformation.
    
    Original Problem: {problem_text}
    Transformed Grid: {transformed_grid_text}
    
    Check if the transformed grid:
    1. Is a valid 2D array
    2. Has dimensions that are reasonable, given input dimensions
    3. Contains values that are consistent with the original grid
    
    Respond with "Valid" or "Invalid".
    """

    validation_result = call_llm(validation_prompt, system_instruction)

    if "Valid" in validation_result:
        return transformed_grid_text
    else:
        return "[[0,0,0],[0,0,0],[0,0,0]]"

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