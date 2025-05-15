import os
import re
import math

# EXPLORATION: Multi-Agent Iterative Refinement with Explicit Rule Validation and Reverse Transformation
# HYPOTHESIS: By introducing multiple specialized agents that iteratively refine the transformation rule and also perform a reverse transformation check,
# we can significantly improve the LLM's ability to generalize grid transformation patterns. This approach leverages explicit rule validation and a novel reverse transformation check to ensure consistency.

def solve_grid_transformation(question, max_attempts=3):
    """Solves grid transformation problems using multi-agent iterative refinement."""

    # Step 1: Rule Extraction by Initial Rule Extractor Agent
    rule_extraction_result = extract_transformation_rule(question)
    if not rule_extraction_result["is_valid"]:
        return f"Error: Initial rule extraction failed. {rule_extraction_result['error']}"
    transformation_rule = rule_extraction_result["transformation_rule"]

    # Step 2: Iterative Rule Refinement by Refinement Agent and Reverse Transformation
    refined_rule_result = refine_transformation_rule(question, transformation_rule, max_attempts=max_attempts)
    if not refined_rule_result["is_valid"]:
        return f"Error: Rule refinement failed. {refined_rule_result['error']}"
    transformation_rule = refined_rule_result["transformation_rule"]

    # Step 3: Apply Transformation by Transformation Agent
    transformed_grid = apply_transformation(question, transformation_rule)
    return transformed_grid

def extract_transformation_rule(question):
    """Extracts the initial transformation rule from the training examples."""
    system_instruction = "You are a highly skilled transformation rule extractor. Your role is to examine training examples and identify the underlying transformation rule clearly and concisely."

    prompt = f"""
    Analyze the following grid transformation problem and extract the underlying transformation rule. Provide the extracted rule in a clear, concise, and easily understandable manner.

    Example 1:
    Problem:
    === TRAINING EXAMPLES ===
    Input Grid:
    [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
    Output Grid:
    [[1, 1, 1], [0, 0, 0], [1, 1, 1]]
    Transformation Rule: Swap the row containing '1' with adjacent rows.

    Problem:
    {question}

    Transformation Rule:
    """

    transformation_rule = call_llm(prompt, system_instruction)

    # Simple validation: Check if the rule is non-empty
    if transformation_rule and transformation_rule.strip():
        return {"is_valid": True, "transformation_rule": transformation_rule, "error": None}
    else:
        return {"is_valid": False, "transformation_rule": None, "error": "Failed to extract a transformation rule."}

def refine_transformation_rule(question, transformation_rule, max_attempts=3):
    """Refines the transformation rule iteratively using a refinement agent and reverse transformation."""
    system_instruction = "You are a transformation rule refinement expert. You will take the given transformation rule and iteratively improve it to ensure accuracy and generalizability."

    for attempt in range(max_attempts):
        prompt = f"""
        You will be provided with a grid transformation problem and a transformation rule. Your task is to critically analyze the rule and refine it to improve its accuracy and generalizability.
        Also, include a "reverse transformation" step to make sure that you can reverse the transformation rule as a means to help with improving correctness.
        Problem: {question}
        Current Transformation Rule: {transformation_rule}

        Example:
        Problem:
        === TRAINING EXAMPLES ===
        Input Grid: [[1, 2], [3, 4]]
        Output Grid: [[4, 3], [2, 1]]
        Current Transformation Rule: Reverse all rows and all columns.
        Refined Transformation Rule: Reverse each row in the grid.

        Refined Transformation Rule:
        """
        refined_transformation_rule = call_llm(prompt, system_instruction)

        # Reverse transformation check (NEW)
        reverse_check_prompt = f"""
        Given a transformation rule, can you provide a reverse transformation?
        Transfomation Rule: {refined_transformation_rule}
        Reverse Transfomation Rule:
        """
        reverse_transformation_rule = call_llm(reverse_check_prompt, system_instruction)

        if refined_transformation_rule and refined_transformation_rule.strip():
            return {"is_valid": True, "transformation_rule": refined_transformation_rule, "error": None}
        else:
            print(f"Attempt {attempt+1}: Refinement failed. Retrying...")
            continue

    return {"is_valid": False, "transformation_rule": None, "error": "Failed to refine transformation rule."}

def apply_transformation(question, transformation_rule):
    """Applies the transformation rule to the test input grid."""
    system_instruction = "You are a highly skilled transformation agent. You will apply a clear transformation rule to the test input grid. Do not deviate and make sure to follow the given rule as best as possible."

    prompt = f"""
    Apply the following transformation rule to the test input grid. Provide ONLY the transformed grid as a list of lists.

    Problem: {question}
    Transformation Rule: {transformation_rule}

    Example:
    Problem:
    Input Grid: [[1, 2], [3, 4]]
    Transformation Rule: Reverse each row.
    Transformed Grid: [[2, 1], [4, 3]]

    Transformed Grid:
    """
    transformed_grid = call_llm(prompt, system_instruction)
    return transformed_grid

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

def main(question):
    """Main function to solve the grid transformation task."""
    try:
        answer = solve_grid_transformation(question)
        return answer
    except Exception as e:
        return f"Error in main function: {str(e)}"