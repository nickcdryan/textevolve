import os
import re
import math

def solve_grid_transformation(question, max_attempts=3):
    """Solves a grid transformation problem using a novel LLM-driven approach with explicit rule extraction and validation."""

    # HYPOTHESIS: The LLM can extract transformation rules more effectively if explicitly prompted to do so AND the extracted rule is validated.
    # This tests a hybrid approach - explicit rule extraction PLUS pattern matching to output a final answer.

    # Step 1: Extract Transformation Rule with validation
    rule_extraction_result = extract_transformation_rule(question, max_attempts=max_attempts)
    if not rule_extraction_result["is_valid"]:
        return f"Error: Could not extract a valid transformation rule. {rule_extraction_result['error']}"
    
    transformation_rule = rule_extraction_result["transformation_rule"]

    # Step 2: Apply Transformation Rule to Test Input.
    transformed_grid = apply_transformation_rule(question, transformation_rule)
    
    # Step 3: Verify that the output transformation is valid.
    output_verification_result = verify_output_grid(question, transformed_grid, transformation_rule, max_attempts=max_attempts)

    if not output_verification_result["is_valid"]:
        return f"Error: Output grid validation failed. {output_verification_result['error']}"

    return transformed_grid

def extract_transformation_rule(question, max_attempts=3):
    """Extracts the transformation rule from the question using LLM with examples and validation."""

    system_instruction = "You are an expert at extracting transformation rules from grid examples."
    
    for attempt in range(max_attempts):
        prompt = f"""
        Given the following grid transformation problem, extract the underlying transformation rule.
        Provide the transformation rule in a concise, human-readable way.

        Example 1:
        Input Grid: [[1, 0], [0, 1]]
        Output Grid: [[2, 0], [0, 2]]
        Transformation Rule: Each '1' is transformed to '2', while '0' remains unchanged.

        Example 2:
        Input Grid: [[1, 2], [3, 4]]
        Output Grid: [[4, 3], [2, 1]]
        Transformation Rule: The grid is rotated 180 degrees.

        Problem:
        {question}

        Transformation Rule:
        """

        transformation_rule = call_llm(prompt, system_instruction)

        # Validation
        validation_prompt = f"""
        Validate the extracted transformation rule for the given problem.
        Check if the rule is complete, consistent with the training examples, and applicable to the test input.

        Problem: {question}
        Extracted Rule: {transformation_rule}

        Is the extracted transformation rule valid? (Answer True/False):
        """

        is_valid = call_llm(validation_prompt, system_instruction)

        if "True" in is_valid:
            return {"is_valid": True, "transformation_rule": transformation_rule, "error": None}
        else:
            error_message = f"Invalid transformation rule (attempt {attempt+1}): {transformation_rule}"
            print(error_message)
            if attempt == max_attempts - 1:
                 return {"is_valid": False, "transformation_rule": None, "error": error_message}

    return {"is_valid": False, "transformation_rule": None, "error": "Failed after multiple attempts."}

def apply_transformation_rule(question, transformation_rule):
    """Applies the extracted transformation rule to the test input using LLM."""
    system_instruction = "You are an expert at applying transformation rules to grids."

    prompt = f"""
    Given the following grid transformation problem and the extracted transformation rule, apply the rule to the test input grid.

    Problem: {question}
    Transformation Rule: {transformation_rule}

    Generate the output grid according to the transformation rule.
    """

    output_grid = call_llm(prompt, system_instruction)
    return output_grid

def verify_output_grid(question, output_grid, transformation_rule, max_attempts=3):
  for attempt in range(max_attempts):
        validation_prompt = f"""
        You are a meticulous grid transformation expert. 
        Problem: {question}
        Transformation Rule: {transformation_rule}
        Output Grid: {output_grid}

        1. Does the output grid follow the transformation rule?
        2. Is the output grid format correct and consistent with the examples in the problem?
        3. Is the output a valid Python list of lists representing the output grid?

        If there are issues, clearly explain what they are. If all checks pass, respond 'VALID'. Otherwise, explain the issues.
        """
        validation_result = call_llm(validation_prompt, system_instruction="You are a meticulous grid transformation expert.")

        if "VALID" in validation_result:
            return {"is_valid": True, "error": None}
        else:
            error_message = f"Validation failed (attempt {attempt + 1}): {validation_result}"
            print(error_message)
            if attempt == max_attempts - 1:
                return {"is_valid": False, "error": error_message}

  return {"is_valid": False, "error": "Failed verification after multiple attempts."}

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

def main(question):
    """Main function to solve the grid transformation task."""
    try:
        answer = solve_grid_transformation(question)
        return answer
    except Exception as e:
        return f"Error in main function: {str(e)}"