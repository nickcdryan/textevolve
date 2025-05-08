import os
import re
import math

# This script solves grid transformation problems using a "Rule Generation and Analogy" approach.
# The hypothesis is that LLMs can generate a transformation rule and then solve using an analogy from the examples.
# This approach focuses on generating a transformation rule *and* finding the closest analogy in training examples.
# A rule is explicitly generated, then an example is chosen whose rule best matches the generated one.

def main(question):
    """Transforms a grid using LLM to generate rules and uses analogy."""
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem by generating rules and using an analogy."""
    system_instruction = "You are an expert at identifying grid transformation patterns from examples and applying them to new grids. You first generate the rule."
    
    # STEP 1: Generate the transformation rule
    rule_generation_prompt = f"""
    You are tasked with identifying the transformation rule applied to grids. Study the training examples and explain the transformation logic in plain English.

    Example 1:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[1, 0], [0, 1]]\n\nOutput Grid:\n[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]\n\n=== TEST INPUT ===\n[[2, 8], [8, 2]]\n\n
    Explanation: Each element in the input grid becomes a diagonal element in the output grid.

    Example 2:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[2, 8], [8, 2]]\n\nOutput Grid:\n[[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]\n\n=== TEST INPUT ===\n[[1, 4], [4, 1]]\n\n
    Explanation: Each element is expanded to a 2x2 block with the element's value.

    Now, explain the transformation rule applied to this example. Respond with ONLY the explanation:
    Test Example:
    {problem_text}
    """
    
    # Attempt to extract the rule
    extracted_rule = call_llm(rule_generation_prompt, system_instruction)
    print(f"Extracted rule: {extracted_rule}")

    # STEP 2: Select the best analogy
    analogy_selection_prompt = f"""
    You are a transformation selector that takes an original problem and the identified rule, then provides the most appropriate example.

    Identified Rule: {extracted_rule}

    Example 1:
    Identified Rule: Each element is expanded to a 2x2 block with the element's value.
    Example transformation problem:\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[2, 8], [8, 2]]\n\nOutput Grid:\n[[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]\n\n=== TEST INPUT ===\n[[1, 4], [4, 1]]\n\n
    Best Example: Each element is expanded to a 2x2 block with the element's value.

    Select the best example that the identified rule best applies to from the original problem. Respond with ONLY the Example and its transformations.
    Test Example:
    {problem_text}
    """
    analogy = call_llm(analogy_selection_prompt, system_instruction)
    print(f"Analogy: {analogy}")
    
    # STEP 3: Transformation Application
    transformation_application_prompt = f"""
    You are an expert in applying grid transformations. The best analogy is provided, apply it to the original problem
    Original Problem: {problem_text}
    Best Analogy: {analogy}

    Example:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[1, 0], [0, 1]]\n\nOutput Grid:\n[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]\n\n=== TEST INPUT ===\n[[2, 8], [8, 2]]\n\n
    Analogy: Each element in the input grid becomes a diagonal element in the output grid.
    Transformed Grid: [[2, 0, 0, 0], [0, 8, 0, 0], [0, 0, 8, 0], [0, 0, 0, 2]]

    Transformed Grid:
    """

    transformed_grid_text = call_llm(transformation_application_prompt, system_instruction)
    print(f"Transformed Grid: {transformed_grid_text}")

    return transformed_grid_text

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