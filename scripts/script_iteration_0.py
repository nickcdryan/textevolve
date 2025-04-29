import os
import re

def main(question):
    """
    Main function to solve the grid transformation task.
    Uses a combination of LLM reasoning and pattern recognition.
    """
    try:
        return solve_grid_transformation(question)
    except Exception as e:
        return f"Error: {str(e)}"

def solve_grid_transformation(question):
    """
    Solves the grid transformation problem by analyzing training examples and applying the learned pattern to the test input.
    Uses multiple LLM calls with embedded examples for robust reasoning and transformation.
    """

    # Step 1: Extract training examples and test input using LLM with multiple examples
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

    Example 2:
    Problem Description:
    Training Examples:
    [{{\"input\":[[0,1],[1,0]],\"output\":[[2,3],[3,2]]}}, {{\"input\":[[1,1],[0,0]],\"output\":[[3,3],[2,2]]}}]
    Test Input:
    [[0,0],[1,1]]
    Extracted Data:
    {{
      "training_examples": "[{{\\"input\\":[[0,1],[1,0]],\\"output\\":[[2,3],[3,2]]}}, {{\"input\":[[1,1],[0,0]],\\"output\\":[[3,3],[2,2]]}}]",
      "test_input": "[[0,0],[1,1]]"
    }}

    Problem Description:
    {question}
    Extracted Data:
    """
    extracted_data_str = call_llm(example_extraction_prompt, "You are an expert at extracting data from text, focusing on the provided format.")
    try:
        extracted_data = eval(extracted_data_str) # Avoid json.loads
        training_examples = extracted_data["training_examples"]
        test_input = extracted_data["test_input"]
    except Exception as e:
        return f"Error extracting data: {str(e)}"

    # Step 2: Analyze the training examples to identify the transformation pattern using LLM with multiple examples
    pattern_analysis_prompt = f"""
    Analyze the training examples to identify the transformation pattern.

    Example 1:
    Training Examples:
    [{{\"input\":[[1,2],[3,4]],\"output\":[[5,6],[7,8]]}}]
    Transformation Pattern:
    Each element in the input grid is increased by 4 to obtain the corresponding element in the output grid.

    Example 2:
    Training Examples:
    [{{\"input\":[[0,1],[1,0]],\"output\":[[2,3],[3,2]]}}, {{\"input\":[[1,1],[0,0]],\"output\":[[3,3],[2,2]]}}]
    Transformation Pattern:
    Each element in the input grid is increased by 2 to obtain the corresponding element in the output grid.

    Training Examples:
    {training_examples}
    Transformation Pattern:
    """
    transformation_pattern = call_llm(pattern_analysis_prompt, "You are an expert at analyzing patterns in data transformations.")

    # Step 3: Apply the transformation pattern to the test input using LLM
    application_prompt = f"""
    Apply the following transformation pattern to the test input.

    Transformation Pattern:
    {transformation_pattern}
    Test Input:
    {test_input}

    Provide the transformed output grid in the same format as the input grid.

    Example:
    Transformation Pattern:
    Each element in the input grid is increased by 4 to obtain the corresponding element in the output grid.
    Test Input:
    [[9,10],[11,12]]
    Transformed Output:
    [[13,14],[15,16]]
    """
    transformed_output = call_llm(application_prompt, "You are an expert at applying transformation patterns to data.")

    # Step 4: Validate the output and handle potential errors
    validation_prompt = f"""
    Validate if the transformed output is in the correct format.
    Transformed Output:
    {transformed_output}

    Example 1:
    Valid Output: [[1,2],[3,4]]
    Example 2:
    Valid Output: [[5, 6, 7], [8, 9, 10]]

    Check if the output is a valid 2D array and has the correct dimensions. If invalid, return "INVALID". If valid, return "VALID".
    """
    validation_result = call_llm(validation_prompt, "You are an expert at validating output formats.")

    if "INVALID" in validation_result:
        return "Error: Invalid output format."

    return transformed_output
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