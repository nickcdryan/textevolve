import os
import re
import math

def main(question):
    """Transforms a grid based on patterns in training examples using LLM-driven iterative refinement with constraint validation."""
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem through iterative rule extraction and application, validated against constraints."""

    system_instruction = "You are an expert at identifying grid transformation patterns from examples, applying them to new grids, and validating the transformed grid against constraints."

    # STEP 1: Extract initial transformation rule with embedded examples
    rule_extraction_prompt = f"""
    You are tasked with identifying transformation rules applied to grids. Study the examples and explain the logic, focusing on spatial relationships and value transformations.

    Example 1:
    Input Grid: [[1, 0], [0, 1]]
    Output Grid: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    Explanation: Each element is expanded diagonally with the element's value.

    Example 2:
    Input Grid: [[2, 8], [8, 2]]
    Output Grid: [[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]
    Explanation: Each element expands to a 2x2 block containing that element.

    Now, explain the transformation rule for this example: {problem_text}
    """

    extracted_rule = call_llm(rule_extraction_prompt, system_instruction)

    # STEP 2: Apply and iteratively refine based on constraint verification
    transformed_grid_text = ""
    for attempt in range(max_attempts):
        application_prompt = f"""
        Transformation Rule: {extracted_rule}
        Apply this rule to: {problem_text}
        Output the transformed grid as a 2D array formatted as a string.

        Example:
        Rule: Double each element
        Input: [[1, 2], [3, 4]]
        Output: [[2, 4], [6, 8]]
        """

        transformed_grid_text = call_llm(application_prompt, system_instruction)

        # Verify constraints with examples
        constraint_verification_prompt = f"""
        Extracted Rule: {extracted_rule}
        Original Grid: {problem_text}
        Transformed Grid: {transformed_grid_text}

        Example 1:
        Rule: Each element is copied diagonally
        Input: [[1,0],[0,1]]
        Transformed: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        Check if the Transformed grid follows the rule's spatial and value constraints. Output 'Yes' or 'No'. Output 'Invalid' if the output is unreadable.
        Result: Yes

        Example 2:
        Rule: Each element doubles.
        Input: [[1,2],[3,4]]
        Transformed: [[1,2],[3,4]]
        Check if the Transformed grid follows the rule's spatial and value constraints. Output 'Yes' or 'No'. Output 'Invalid' if the output is unreadable.
        Result: No

        Check if the Transformed Grid follows the rule's spatial and value constraints. Output 'Yes' or 'No'. Output 'Invalid' if the output is unreadable.
        """

        verification_result = call_llm(constraint_verification_prompt, system_instruction)

        if "Yes" in verification_result and "[" in transformed_grid_text and "]" in transformed_grid_text:
            return transformed_grid_text
        else:
            # Refine the extracted rule based on feedback
            refinement_prompt = f"""
            The transformation rule or generated grid failed validation. Review the original problem, extracted rule, and generated grid, then refine the rule.

            Original Problem: {problem_text}
            Extracted Rule: {extracted_rule}
            Generated Grid: {transformed_grid_text}
            Validation Result: {verification_result}

            Provide a refined explanation of the rule focusing on spatial relationships, value transformations, constraints:
            """

            extracted_rule = call_llm(refinement_prompt, system_instruction)
            print(f"Attempt {attempt+1} failed, refining rule: {extracted_rule}")

    return "[[0,0,0],[0,0,0],[0,0,0]]" # Fallback after max attempts

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