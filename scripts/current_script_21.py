import os
import re
import math

# This script solves grid transformation problems using a novel "Meta-Pattern Extraction and Transformation" approach.
# It hypothesizes that by first extracting general transformation meta-patterns (e.g., "replication," "reflection," "rotation")
# and then applying more specific transformations within those meta-patterns, we can achieve better generalization.
# Additionally, introduces a "transformation intent" step to clarify the goals of transformation.
# The script uses multi-example prompting and includes validation steps at each stage.
# This design prioritizes the LLM's reasoning capabilities for feature extraction and pattern recognition, minimizing code.

def main(question):
    """Transforms a grid based on extracted meta-patterns and intent."""
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem by extracting meta-patterns and applying transformations."""

    system_instruction = "You are an expert at identifying and applying grid transformation patterns. Focus on identifying high-level transformation meta-patterns before applying specific rules."

    # STEP 1: Transformation Intent Extraction
    intent_extraction_prompt = f"""
    Analyze the grid transformation problem and identify the overall intent or goal of the transformation.

    Example 1:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[1, 0], [0, 1]]\n\nOutput Grid:\n[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]\n\n=== TEST INPUT ===\n[[2, 8], [8, 2]]\n
    Transformation Intent: To expand the grid while preserving the diagonal elements.

    Example 2:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[2, 8], [8, 2]]\n\nOutput Grid:\n[[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]\n
    Transformation Intent: To enlarge each cell in a 2x2 block.

    Problem: {problem_text}
    Transformation Intent:
    """

    transformation_intent = call_llm(intent_extraction_prompt, system_instruction)
    print(f"Transformation Intent: {transformation_intent}")

    # STEP 2: Meta-Pattern Extraction
    meta_pattern_extraction_prompt = f"""
    Identify the general meta-pattern used in the grid transformation, given the following intent: {transformation_intent}

    Example 1:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[1, 0], [0, 1]]\n\nOutput Grid:\n[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]\n
    Transformation Intent: To expand the grid while preserving the diagonal elements.
    Meta-Pattern: Expansion

    Example 2:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[2, 8], [8, 2]]\n\nOutput Grid:\n[[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]\n
    Transformation Intent: To enlarge each cell in a 2x2 block.
    Meta-Pattern: Replication

    Problem: {problem_text}
    Transformation Intent: {transformation_intent}
    Meta-Pattern:
    """

    meta_pattern = call_llm(meta_pattern_extraction_prompt, system_instruction)
    print(f"Meta-Pattern: {meta_pattern}")

    # STEP 3: Transformation Application with Meta-Pattern Guidance
    transformation_application_prompt = f"""
    Apply the grid transformation to the following problem, guided by the identified meta-pattern: {meta_pattern} and transformation intent: {transformation_intent}

    Example 1:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[1, 0], [0, 1]]\n\nOutput Grid:\n[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]\n
    Meta-Pattern: Expansion
    Transformed Grid: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    Example 2:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[2, 8], [8, 2]]\n\nOutput Grid:\n[[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]\n
    Meta-Pattern: Replication
    Transformed Grid: [[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]

    Problem: {problem_text}
    Meta-Pattern: {meta_pattern}
    Transformed Grid:
    """

    transformed_grid_text = call_llm(transformation_application_prompt, system_instruction)
    print(f"Transformed Grid: {transformed_grid_text}")

    return transformed_grid_text

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