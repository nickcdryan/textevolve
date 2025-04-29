import os
import re
import math

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

def analyze_grid_transformation(question, max_attempts=3):
    """Analyzes the grid transformation task and applies transformation logic."""

    # Hypothesis: By analyzing training examples to identify key transformation rules (e.g., proximity to specific values), we can apply them to test inputs. This new approach focuses on identifying specific number proximity rules rather than a more generic structural approach. Also, uses iterative refinement with specific validators.

    # Step 1: Extract training examples and test input using LLM
    extraction_prompt = f"""
    Extract training examples and test input from this grid transformation question.

    Example Input:
    Grid Transformation Task

    Training Examples:
    [
        {{"input":[[0,0,0],[0,1,0],[0,0,0]],"output":[[1,1,1],[1,1,1],[1,1,1]]}},
        {{"input":[[0,0,1],[0,0,0],[0,0,0]],"output":[[0,0,0],[0,0,0],[1,1,1]]}}
    ]

    Test Input:
    [[0,0,0],[0,1,0],[0,0,1]]

    Your Extraction:
    """

    extracted_content = call_llm(question + extraction_prompt, "You are an extraction specialist focusing on structured data.")

    training_examples = extracted_content.split("Test Input:")[0].split("Training Examples:")[1].strip()
    test_input = extracted_content.split("Test Input:")[1].strip()

    # Step 2: LLM identifies transformation rules from training examples.

    rule_identification_prompt = f"""
    Identify the transformation rules from these training examples (as plain text) and test grid to transform to based on the transformation rules (using same grid format)
    Training Examples:
    {training_examples}

    Test Grid:
    {test_input}

    Transformation Rules:
    """

    transformation_rules = call_llm(rule_identification_prompt, "You are an expert data scientist who specializes in identifying patterns")

    # Step 3: Apply transformation rules to the test input and generate solution.

    transformation_prompt = f"""
    Apply the transformation rules in plain text to the test grid (using same grid format).

    Test Grid:
    {test_input}

    Transformation Rules:
    {transformation_rules}

    Transformed Test Grid:
    """
    transformed_grid = call_llm(transformation_prompt, "You are an expert grid transformation specialist.")

    # Verification step:
    verification_prompt = f"""
    Verify that the transformation correctly follows the transformation rules.

    Original Test Grid:
    {test_input}

    Transformation Rules:
    {transformation_rules}

    Transformed Grid:
    {transformed_grid}

    Is the transformation valid? Respond with VALID or INVALID. If invalid, explain why.
    """
    verification_result = call_llm(verification_prompt, "You are a grid transformation validator.")

    if "INVALID" in verification_result:
        print("Transformation validation failed. Retrying with refined rules.")

        refine_prompt = f"""
        The previous transformation was found to be invalid with reason: {verification_result}

        Based on that, refine the transformation rules to be less erroneous.

        Original Test Grid:
        {test_input}

        Transformation Rules:
        {transformation_rules}

        Provide refined transformation rules:
        """

        refined_rules = call_llm(refine_prompt, "You are an expert in refining transformation rules to eliminate errors.")

        transformation_prompt = f"""
        Apply the transformation rules in plain text to the test grid (using same grid format).

        Test Grid:
        {test_input}

        Transformation Rules:
        {refined_rules}

        Transformed Test Grid:
        """
        transformed_grid = call_llm(transformation_prompt, "You are an expert grid transformation specialist.")
    return transformed_grid
def main(question):
    """Main function to process the question."""
    try:
        result = analyze_grid_transformation(question)
        return result
    except Exception as e:
        return f"Error: {str(e)}"