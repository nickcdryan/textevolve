import os
import re

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
    """Analyzes grid transformation problems using a novel LLM-driven approach that uses multiple examples in a more structured way."""

    # This approach focuses on using the LLM to directly transform the input grid by learning from examples.
    # Hypothesis: Providing multiple examples in a specific format can enable the LLM to learn a transformation rule implicitly and apply it effectively.

    # Step 1: Create a multi-example prompt for direct grid transformation
    examples = """
    Example 1:
    Input Grid:
    [[0, 2, 2, 0, 0], [0, 2, 2, 0, 0], [0, 0, 0, 0, 4], [0, 0, 0, 0, 4], [0, 0, 0, 0, 0]]
    Output Grid:
    [[0, 0, 0, 0, 0], [0, 2, 2, 0, 4], [0, 2, 2, 0, 4], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

    Example 2:
    Input Grid:
    [[0, 0, 0, 4, 4], [0, 0, 0, 4, 4], [0, 2, 2, 0, 0], [0, 2, 2, 0, 0], [0, 0, 0, 0, 0]]
    Output Grid:
    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 2, 2, 4, 4], [0, 2, 2, 4, 4], [0, 0, 0, 0, 0]]

    Example 3:
    Input Grid:
    [[0, 0, 0, 0, 0], [0, 0, 0, 2, 0], [0, 1, 0, 2, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]
    Output Grid:
    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 2, 0], [0, 1, 0, 2, 0], [0, 0, 0, 0, 0]]
    """

    transformation_prompt = f"""
    You are an expert at transforming grids based on learned patterns.
    Analyze the following examples and apply the learned transformation rule to the test input grid.
    Return ONLY the transformed grid as a list of lists.
    DO NOT explain the rule.

    {examples}

    Test Input Grid:
    {question}

    Transformed Grid:
    """

    # Step 2: Call the LLM to directly transform the test input
    transformed_grid = call_llm(transformation_prompt, system_instruction="You are a grid transformation expert.")

    # Step 3: Implement a verification step to check if the output is a valid list of lists
    verification_prompt = f"""
    You are a verifier. Verify that the output is a valid Python list of lists representing a grid.
    If valid, return VALID. If invalid, return INVALID with a brief explanation.

    Output:
    {transformed_grid}

    Verification:
    """

    verification_result = call_llm(verification_prompt, system_instruction="You are a verifier that verifies that the output is a valid Python list of lists.")

    if "INVALID" in verification_result:
        return f"Error: Invalid output format. {verification_result}"

    # Step 4: Implement another verifier to ensure the output matches the test input grid dimensions

    dimension_check_prompt = f"""
        You are a dimension checker.

        Test Input Grid:
        {question}

        Transformed Grid:
        {transformed_grid}

        Verify if the dimensions are different. Provide either 'SAME' or 'DIFFERENT' and a very short summary
        """

    dimension_check = call_llm(dimension_check_prompt, system_instruction="You are a dimension expert.")

    return transformed_grid

def main(question):
    """Main function to process the grid transformation question."""
    try:
        answer = analyze_grid_transformation(question)
        return answer
    except Exception as e:
        return f"Error: {str(e)}"