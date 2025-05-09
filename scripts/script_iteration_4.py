import os
import re
import math

def main(question):
    """
    Transforms a grid based on patterns in training examples using LLM-driven localized pattern reinforcement.
    Uses localized pattern identification, reinforcement, and a feedback loop to improve pattern generalization.
    """
    return solve_grid_transformation(question)

def solve_grid_transformation(problem, max_attempts=3):
    """Solve grid transformation problems by identifying, reinforcing, and applying localized patterns."""

    # Hypothesis: Reinforcing localized patterns through a feedback loop will improve generalization.
    system_instruction = "You are an expert at identifying localized patterns and generalizing them to grid transformations."

    # Step 1: Extract training examples and the test input grid.
    extraction_prompt = f"""
    Extract the training examples and the test input grid from the problem description.

    Example:
    Problem: Grid Transformation Task... Input Grid: [[1,2],[3,4]] ... Output Grid: [[5,6],[7,8]] ... TEST INPUT: [[9,10],[11,12]]
    Extracted: {{"examples": ["Input Grid: [[1,2],[3,4]] ... Output Grid: [[5,6],[7,8]]"], "test_input": "[[9,10],[11,12]]"}}

    Problem: {problem}
    Extracted:
    """
    extracted_info = call_llm(extraction_prompt, system_instruction)
    print(f"Extracted Info: {extracted_info}")

    # Step 2: Identify localized patterns.
    pattern_identification_prompt = f"""
    Identify localized patterns in the training examples.

    Example:
    Examples: Input Grid: [[1, 0], [0, 1]] ... Output Grid: [[2, 0], [0, 2]]
    Localized Pattern: If a cell has value 1, transform it to 2.

    Examples: {extracted_info}
    Localized Pattern:
    """
    localized_patterns = call_llm(pattern_identification_prompt, system_instruction)
    print(f"Localized Patterns: {localized_patterns}")

    # Step 3: Reinforce the identified patterns with feedback.

    reinforcement_prompt = f"""
    Reinforce the following identified localized patterns by providing more precise and detailed rules, addressing potential edge cases.

    Patterns: {localized_patterns}
    Examples: {extracted_info}

    Example 1:
    Patterns: If a cell has value 1, transform it to 2.
    Reinforced Patterns: If a cell has value 1, transform it to 2 only if adjacent cells do not have value 8.

    Reinforced Patterns:
    """
    reinforced_patterns = call_llm(reinforcement_prompt, system_instruction)
    print(f"Reinforced Patterns: {reinforced_patterns}")

    # Step 4: Apply the reinforced patterns to the test input grid.
    transformation_prompt = f"""
    Apply the reinforced localized patterns to transform the test input grid.

    Reinforced Patterns: {reinforced_patterns}
    Test Input Grid: {extracted_info}

    Example:
    Patterns: If a cell has value 1, transform it to 2. Test Input Grid: [[1, 0], [0, 1]]
    Transformed Grid: [[2, 0], [0, 2]]

    Transformed Grid:
    """
    transformed_grid = call_llm(transformation_prompt, system_instruction)
    print(f"Transformed Grid: {transformed_grid}")

    # Step 5: Verify the transformed grid with feedback and iterate if needed.
    verification_prompt = f"""
    Verify the transformed grid based on the reinforced localized patterns and training examples. Provide specific feedback if there are errors.

    Reinforced Patterns: {reinforced_patterns}
    Test Input Grid: {extracted_info}
    Transformed Grid: {transformed_grid}

    Example:
    Patterns: If a cell has value 1, transform it to 2. Input: [[1,0],[0,1]]. Output: [[2,0],[0,2]]. Verification: CORRECT.
    Patterns: If cell =8, set neighbours to 4. Input: [[8,0],[0,1]]. Output: [[8,4],[4,1]]. Verification: CORRECT.

    Verification: Does the transformed grid follow the reinforced localized patterns? Answer 'yes' or 'no' with specific details.
    """
    verification_result = call_llm(verification_prompt, system_instruction)
    print(f"Verification Result: {verification_result}")

    if "yes" in verification_result.lower():
        return transformed_grid
    else:
        return "Unable to transform the grid correctly."

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