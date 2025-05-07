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
    """Analyzes grid transformation problems using multiple LLM calls."""

    # Step 1: Rule Extraction with Few-Shot Examples
    extraction_prompt = f"""
    Analyze the following grid transformation problem and extract the underlying rule.

    Example 1:
    Input Grid:
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    Output Grid:
    [[1, 1], [1, 1]]
    Rule: The output grid is a 2x2 grid using the top-left corner of the input.

    Example 2:
    Input Grid:
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    Output Grid:
    [[3, 2, 1], [6, 5, 4], [9, 8, 7]]
    Rule: The output grid is the reverse of the input grid.

    Problem:
    {question}

    Extracted Rule:
    """

    extracted_rule = call_llm(extraction_prompt, system_instruction="You are an expert at extracting rules from grid transformations.")

    # Step 2: Rule Verification
    verification_prompt = f"""
    Verify if the extracted rule is correct based on the problem description.

    Problem:
    {question}
    Extracted Rule:
    {extracted_rule}

    Is the rule valid? (Yes/No):
    """
    is_rule_valid = call_llm(verification_prompt, system_instruction="You are a rule verification expert.")

    if "Yes" not in is_rule_valid:
        return "Error: Invalid rule extracted."

    # Step 3: Rule Application
    application_prompt = f"""
    Apply the following rule to the test input to generate the output grid.

    Rule:
    {extracted_rule}
    Test Input:
    {question}

    Output Grid:
    """

    output_grid = call_llm(application_prompt, system_instruction="You are an expert at applying rules to grid transformations.")

    # Step 4: Formatting
    formatting_prompt = f"""
    Format the output grid as a list of lists.

    Input Grid:
    {output_grid}
    Formatted Grid:
    """
    formatted_grid = call_llm(formatting_prompt, system_instruction="You are a formatting expert.")

    return formatted_grid

def main(question):
    """Main function to process the grid transformation question."""
    try:
        answer = analyze_grid_transformation(question)
        return answer
    except Exception as e:
        return f"Error: {str(e)}"