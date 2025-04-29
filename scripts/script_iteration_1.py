import os
import re

def main(question):
    """
    Main function to solve the grid transformation task.
    This approach focuses on describing the target grid rather than directly transforming it.
    """
    try:
        return solve_grid_transformation(question)
    except Exception as e:
        return f"Error: {str(e)}"

def solve_grid_transformation(question):
    """
    Solves the grid transformation problem by describing the output grid based on training examples.
    """

    # Step 1: Extract training examples and test input using LLM
    example_extraction_prompt = f"""
    Extract the training examples and test input from the following problem description.

    Example 1:
    Problem Description:
    Training Examples:
    [{{\"input\":[[1,2],[3,4]],\"output\":[[5,6],[7,8]]}}]
    Test Input:
    [[9,10],[11,12]]
    Extracted Data:
    {{
      "training_examples": "[{{\\"input\\":[[1,2],[3,4]],\\"output\\":[[5,6],[7,8]]}}]",
      "test_input": "[[9,10],[11,12]]"
    }}

    Problem Description:
    {question}
    Extracted Data:
    """
    extracted_data_str = call_llm(example_extraction_prompt, "You are an expert at extracting data from text.")

    # Basic string parsing to extract data (avoiding json.loads)
    try:
        training_examples = re.search(r'"training_examples":\s*"([^"]*)"', extracted_data_str).group(1)
        test_input = re.search(r'"test_input":\s*"([^"]*)"', extracted_data_str).group(1)
    except Exception as e:
        return f"Error extracting data: {str(e)}"

    # Step 2: Describe the transformation target grid
    target_description_prompt = f"""
    Describe the characteristics of the target grid based on training examples. Focus on how the output grid differs from a base grid (all zeroes).
    Example 1:
    Training Examples:
    [{{\"input\":[[0,0],[0,0]],\"output\":[[1,1],[1,1]]}}]
    Description: The output grid is the same size as the input grid. Every non-zero element in the output grid is set to the value of 1.
    Example 2:
    Training Examples:
    [{{\"input\":[[1,2],[3,4]],\"output\":[[5,6],[7,8]]}}]
    Description: The output grid is the same size as the input grid. Each element is the sum of the input matrix element and 4.
    Training Examples:
    {training_examples}
    Description:
    """
    target_description = call_llm(target_description_prompt, "You are an expert at describing grid transformations.")

    # Step 3: Apply the description to the test input
    transformation_prompt = f"""
    Apply the following description to the test input. Generate the transformed output grid.

    Description:
    {target_description}
    Test Input:
    {test_input}

    Example:
    Description:
    The output grid is the same size as the input grid. Every non-zero element in the output grid is set to the value of 1.
    Test Input:
    [[0,0],[0,0]]
    Transformed Output:
    [[1,1],[1,1]]
    """
    transformed_output = call_llm(transformation_prompt, "You are an expert at applying descriptions to generate transformed grids.")

    # Step 4: Validation (Check if the response has the correct formatting)
    validation_prompt = f"""
    Validate if the transformed output is in the correct format.
    Transformed Output:
    {transformed_output}
    Is the output a valid 2D array (e.g. [[1,2],[3,4]])? Answer "VALID" or "INVALID".
    """
    validation_result = call_llm(validation_prompt, "You are an expert at validating output formats.")

    if "INVALID" in validation_result:
        return "Error: Invalid output format."

    return transformed_output

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt."""
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