import os
import re
import math

def main(question):
    """
    Solves grid transformation tasks by a) analyzing the training examples to categorize the transformation type,
    b) applying a transformation-specific strategy (different from previous approaches), and c) verifying the result.

    This approach tests the hypothesis that categorizing transformations upfront enables more targeted and successful application.
    It also includes detailed examples within each LLM prompt to improve reliability and uses verification to check results at different parts of the pipeline.
    """
    try:
        # Step 1: Categorize transformation type (NEW STEP - tests the core hypothesis)
        transformation_type = categorize_transformation(question)
        if "Error" in transformation_type:
            return f"Transformation categorization failed: {transformation_type}"

        # Step 2: Apply transformation based on its type, using the appropriate function
        if transformation_type == "Reflection":
            transformed_grid = apply_reflection(question)
        elif transformation_type == "Replication":
            transformed_grid = apply_replication(question)
        elif transformation_type == "Arithmetic":
            transformed_grid = apply_arithmetic(question)
        else:
            return "Unsupported transformation type."

        # Step 3: Verify the transformed grid
        verification_result = verify_transformation(question, transformed_grid)
        if not verification_result["is_valid"]:
            return f"Transformation failed verification: {verification_result['feedback']}"

        return transformed_grid

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def categorize_transformation(question):
    """Categorizes the type of transformation using LLM analysis."""
    system_instruction = "You are an expert at categorizing grid transformations."
    prompt = f"""
    Analyze the following question and categorize the transformation type into one of these categories:
    Reflection, Replication, or Arithmetic.

    Example 1:
    Question: Grid Transformation Task Training Examples: [{{'input': [[0, 1], [1, 0]], 'output': [[1, 0], [0, 1]]}}] Test Input: [[4,5],[5,4]]
    Category: Reflection

    Example 2:
    Question: Grid Transformation Task Training Examples: [{{'input': [[1]], 'output': [[1, 1], [1, 1]]}}] Test Input: [[2]]
    Category: Replication

    Example 3:
    Question: Grid Transformation Task Training Examples: [{{'input': [[1]], 'output': [[2]]}}] Test Input: [[3]]
    Category: Arithmetic

    Question: {question}
    Category:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error categorizing transformation: {str(e)}"

def apply_reflection(question):
    """Applies reflection transformation."""
    system_instruction = "You are an expert at applying reflection transformations to grids."
    prompt = f"""
    Apply the reflection transformation to the test input based on the training examples.

    Example:
    Question: Grid Transformation Task Training Examples: [{{'input': [[0, 1], [1, 0]], 'output': [[1, 0], [0, 1]]}}] Test Input: [[4,5],[5,4]]
    Transformed Grid: [[5, 4], [4, 5]]

    Question: {question}
    Transformed Grid:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error applying reflection: {str(e)}"

def apply_replication(question):
    """Applies replication transformation."""
    system_instruction = "You are an expert at applying replication transformations to grids."
    prompt = f"""
    Apply the replication transformation to the test input based on the training examples.

    Example:
    Question: Grid Transformation Task Training Examples: [{{'input': [[1]], 'output': [[1, 1], [1, 1]]}}] Test Input: [[2]]
    Transformed Grid: [[2, 2], [2, 2]]

    Question: {question}
    Transformed Grid:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error applying replication: {str(e)}"

def apply_arithmetic(question):
    """Applies arithmetic transformation."""
    system_instruction = "You are an expert at applying arithmetic transformations to grids."
    prompt = f"""
    Apply the arithmetic transformation to the test input based on the training examples.

    Example:
    Question: Grid Transformation Task Training Examples: [{{'input': [[1]], 'output': [[2]]}}] Test Input: [[3]]
    Transformed Grid: [[4]]

    Question: {question}
    Transformed Grid:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error applying arithmetic: {str(e)}"

def verify_transformation(question, transformed_grid):
    """Verifies if the transformed grid is correct."""
    system_instruction = "You are an expert at verifying grid transformations."
    prompt = f"""
    Verify if the transformed grid is correct based on the training examples in the question.

    Example:
    Question: Grid Transformation Task Training Examples: [{{'input': [[0, 1], [1, 0]], 'output': [[1, 0], [0, 1]]}}] Test Input: [[4,5],[5,4]]
    Transformed Grid: [[5, 4], [4, 5]]
    Verification: {{"is_valid": true, "feedback": "The transformation transposes the input grid."}}

    Question: {question}
    Transformed Grid: {transformed_grid}
    Verification:
    """
    try:
        verification_result = call_llm(prompt, system_instruction)
        # Simple validity check.
        if "true" in verification_result.lower():
            return {"is_valid": True, "feedback": "Transformation is valid."}
        else:
            return {"is_valid": False, "feedback": verification_result}
    except Exception as e:
        return {"is_valid": False, "feedback": f"Error verifying transformation: {str(e)}"}

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