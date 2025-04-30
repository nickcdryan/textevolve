def main(question):
    """
    Solves grid transformation tasks by analyzing training examples and applying the learned transformation to a test input.
    Leverages LLM for pattern recognition and transformation.
    """
    try:
        # Extract training examples and test input
        training_examples_str = question.split("Training Examples:\n")[1].split("\n\nTest Input:")[0]
        test_input_str = question.split("Test Input:\n")[1].split("\n\nTransform")[0]

        # Analyze the transformation pattern using LLM
        pattern_description = analyze_transformation_pattern(training_examples_str)

        # Apply the transformation pattern to the test input using LLM
        transformed_grid = apply_transformation(test_input_str, pattern_description)

        return transformed_grid

    except Exception as e:
        return f"Error: {str(e)}"

def analyze_transformation_pattern(training_examples_str):
    """
    Analyzes training examples to identify the transformation pattern.
    Uses LLM with chain-of-thought reasoning and embedded examples for robust pattern extraction.
    """
    system_instruction = "You are an expert at identifying transformation patterns in grid data."
    prompt = f"""
    Analyze the following training examples and describe the transformation pattern in a concise, step-by-step manner.

    Example 1:
    Training Examples:
    [
        {{"input": [[0, 0, 1], [0, 0, 0], [0, 0, 0]], "output": [[1, 1, 1], [0, 0, 0], [0, 0, 0]]}},
        {{"input": [[0, 2, 0], [0, 0, 0], [0, 0, 0]], "output": [[2, 2, 2], [0, 0, 0], [0, 0, 0]]}}
    ]
    Transformation Pattern:
    1. Identify the non-zero value in the input grid.
    2. Replace all values in the first row of the output grid with that non-zero value.
    3. Keep all other rows as zero.

    Example 2:
    Training Examples:
    [
        {{"input": [[0, 0, 0], [0, 1, 0], [0, 0, 0]], "output": [[0, 0, 0], [1, 1, 1], [0, 0, 0]]}},
        {{"input": [[0, 0, 0], [2, 0, 0], [0, 0, 0]], "output": [[2, 2, 2], [2, 2, 2], [2, 2, 2]]}}
    ]
    Transformation Pattern:
    1. Replace all values in the output grid with the first non-zero value found in the input grid.
        

    Training Examples:
    {training_examples_str}
    Transformation Pattern:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error analyzing transformation pattern: {str(e)}"

def apply_transformation(test_input_str, pattern_description):
    """
    Applies the transformation pattern to the test input.
    Uses LLM with chain-of-thought reasoning and embedded examples to generate the transformed grid.
    """
    system_instruction = "You are an expert at applying transformation patterns to grid data."
    prompt = f"""
    Apply the following transformation pattern to the given test input.

    Example 1:
    Test Input:
    [[0, 0, 0], [0, 5, 0], [0, 0, 0]]
    Transformation Pattern:
    1. Identify the non-zero value in the input grid.
    2. Replace all values in the middle row of the output grid with that non-zero value.
    3. Keep all other rows as zero.
    Transformed Grid:
    [[0, 0, 0], [5, 5, 5], [0, 0, 0]]

    Example 2:
    Test Input:
    [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
    Transformation Pattern:
    1. Multiply each value by 2 in the input grid
    Transformed Grid:
    [[2, 0, 0], [0, 0, 0], [0, 0, 0]]

    Test Input:
    {test_input_str}
    Transformation Pattern:
    {pattern_description}
    Transformed Grid:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error applying transformation: {str(e)}"

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response."""
    try:
        import os
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