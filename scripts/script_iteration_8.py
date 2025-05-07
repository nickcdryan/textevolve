import os
import re

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

def analyze_grid_transformation(question, max_attempts=3):
    """Analyzes grid transformation problems using multiple LLM calls with enhanced rule extraction and verification."""

    # Step 1: Enhanced Rule Extraction with More Detailed Few-Shot Examples
    extraction_prompt = f"""
    Analyze the following grid transformation problem and extract the underlying rule in detail.

    Example 1:
    Input Grid:
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    Output Grid:
    [[1, 1], [1, 1]]
    Rule: The output grid is a 2x2 grid using the top-left corner of the input. The dimensions are reduced and the top-left section is preserved.

    Example 2:
    Input Grid:
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    Output Grid:
    [[3, 2, 1], [6, 5, 4], [9, 8, 7]]
    Rule: The output grid is the reverse of the input grid. Each row is reversed individually.

    Example 3:
    Input Grid:
    [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 0, 1, 0], [0, 2, 2, 0, 1, 0, 0, 0, 1, 0], [0, 2, 2, 0, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 1, 1, 0, 1, 1, 0], [0, 2, 0, 0, 0, 1, 0, 0, 1, 0], [0, 2, 0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]
    Output Grid:
    [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 0, 1, 0], [0, 2, 0, 0, 1, 0, 0, 0, 1, 0], [0, 2, 2, 0, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 1, 1, 0, 1, 1, 0], [0, 2, 0, 0, 0, 1, 0, 0, 1, 0], [0, 2, 0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]
    Rule: Change some 0's to 2's by checking the adjacent numbers.

    Problem:
    {question}

    Extracted Rule:
    """

    extracted_rule = call_llm(extraction_prompt, system_instruction="You are an expert at extracting rules from grid transformations. Provide as much detail as possible, such as the specific number changes based on positions and adjacent numbers.")

    # Step 2: Enhanced Rule Verification with Reasoning
    verification_prompt = f"""
    Verify if the extracted rule is correct based on the problem description and provide a detailed explanation.

    Problem:
    {question}
    Extracted Rule:
    {extracted_rule}

    Is the rule valid? (Yes/No). Explain your reasoning:
    """
    is_rule_valid = call_llm(verification_prompt, system_instruction="You are a rule verification expert. Explain why you believe the extracted rule is correct or incorrect.")

    if "Yes" not in is_rule_valid:
        return "Error: Invalid rule extracted. " + is_rule_valid

    # Step 3: Rule Application with context
    application_prompt = f"""
    Apply the following rule to the test input to generate the output grid. Return the grid as a list of lists.

    Rule:
    {extracted_rule}
    Test Input:
    {question}

    Output Grid:
    """

    output_grid = call_llm(application_prompt, system_instruction="You are an expert at applying rules to grid transformations. Ensure the output is a valid grid.")

    # Step 4: Grid Verification to ensure a valid output
    grid_verification_prompt = f"""
    Verify that this grid output is in the correct list of lists format.
    {output_grid}

    Return the output as a list of lists.
    """
    formatted_grid = call_llm(grid_verification_prompt, system_instruction="Return the grid as a list of lists")

    return formatted_grid

def main(question):
    """Main function to process the grid transformation question."""
    try:
        answer = analyze_grid_transformation(question)
        return answer
    except Exception as e:
        return f"Error: {str(e)}"