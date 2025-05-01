import os
import re
import math

def main(question):
    """Transforms a grid based on patterns in training examples using a "Spatial Relationship Analysis and Rule-Based Synthesis" approach.

    Hypothesis: By explicitly analyzing the spatial relationships between elements and encoding these relationships into structured rules, we can improve the LLM's ability to generalize transformations, especially when dimensions vary. This approach uses an intermediate structured rule representation. It is different because it uses a spatial relationship extractor, a dimension predictor, and a rule application synthesizer.

    This approach uses a spatial relationship extractor, a dimension predictor, and a rule application synthesizer. We focus on structured rules based on spatial relationships between elements.
    """
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem by analyzing spatial relationships and synthesizing a new grid."""

    system_instruction = "You are a master of spatial reasoning and grid transformations. Analyze spatial relationships and synthesize grids according to structured rules."

    # STEP 1: Extract Spatial Relationships - with examples!
    spatial_relationship_prompt = f"""
    Analyze the training examples and extract spatial relationships between elements. Focus on relationships such as:
    - Element position (row, column)
    - Relative position to other elements (above, below, left, right)
    - Value-based conditions (if element X is present, element Y changes)

    Example 1:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[1, 0], [0, 1]]\n\nOutput Grid:\n[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]\n
    Spatial Relationships:
    - The input grid's '1' elements become diagonals in the output grid. The output grid is an expansion of the input grid.

    Example 2:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[2, 8], [8, 2]]\n\nOutput Grid:\n[[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]\n
    Spatial Relationships:
    - Each element in the input grid is expanded into a 2x2 block in the output grid. The output grid is an expansion of the input grid.

    Problem: {problem_text}
    Spatial Relationships:
    """

    extracted_relationships = call_llm(spatial_relationship_prompt, system_instruction)
    print(f"Extracted Relationships: {extracted_relationships}") # Diagnostic

    # STEP 2: Predict Output Grid Dimensions - with examples!
    dimension_prediction_prompt = f"""
    Predict the dimensions of the output grid based on the spatial relationships and the input grid dimensions. Explain the rule to determine the dimensions of the output grid.

    Example 1:
    Spatial Relationships: The input grid's '1' elements become diagonals in the output grid. The output grid is an expansion of the input grid.
    Input Grid Dimensions: 2x2
    Predicted Output Dimensions: 4x4 (Each dimension is doubled). Explanation: Since each element becomes a diagonal, it doubles the size in rows and columns.

    Example 2:
    Spatial Relationships: Each element in the input grid is expanded into a 2x2 block in the output grid. The output grid is an expansion of the input grid.
    Input Grid Dimensions: 2x2
    Predicted Output Dimensions: 4x4 (Each dimension is doubled). Explanation: Since each element is expanded into 2x2 blocks, it doubles the size in both rows and columns.

    Spatial Relationships: {extracted_relationships}
    Input Grid: {problem_text}
    Predicted Output Dimensions:
    """

    predicted_dimensions = call_llm(dimension_prediction_prompt, system_instruction)
    print(f"Predicted Dimensions: {predicted_dimensions}") # Diagnostic

    #STEP 3: Rule-Based Synthesis
    synthesis_prompt = f"""
    Synthesize the output grid based on the extracted spatial relationships and the predicted output dimensions. Provide the output grid as a 2D array formatted as a string, WITHOUT any additional explanation or comments.

    Spatial Relationships: {extracted_relationships}
    Output Dimensions: {predicted_dimensions}
    Input Grid: {problem_text}

    Example:
    Spatial Relationships: Each element in the input grid is expanded into a 2x2 block in the output grid.
    Output Dimensions: 4x4
    Input Grid: [[1, 2], [3, 4]]
    Transformed Grid: [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]

    Transformed Grid:
    """

    for attempt in range(max_attempts):
        try:
            transformed_grid_text = call_llm(synthesis_prompt, system_instruction)
            # Basic validation - check if it looks like a grid
            if "[" in transformed_grid_text and "]" in transformed_grid_text:
                return transformed_grid_text
            else:
                print(f"Attempt {attempt+1} failed: Output does not resemble a grid. Retrying...")
        except Exception as e:
            print(f"Attempt {attempt+1} failed with error: {e}. Retrying...")

    # Fallback approach if all attempts fail
    return "[[0,0,0],[0,0,0],[0,0,0]]"


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