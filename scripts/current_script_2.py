import os
import re

def main(question):
    """
    Transform a grid based on patterns in training examples using an LLM.
    """

    # Formulate the prompt with multiple examples to guide the LLM
    prompt = f"""
    You are an expert grid transformer. Analyze the training examples below to
    identify the transformation rule and apply it to the test input.

    Example 1:
    Input Grid:
    [[0, 0, 8, 0, 0], [0, 0, 8, 0, 0], [8, 8, 8, 8, 8], [0, 0, 8, 2, 2], [0, 0, 8, 2, 2]]
    Output Grid:
    [[0, 0, 8, 0, 0], [0, 0, 8, 0, 0], [8, 8, 8, 8, 8], [0, 0, 8, 2, 2], [0, 0, 8, 2, 2]]
    Reasoning: No transformation is apparent. The output is identical to the input.

    Example 2:
    Input Grid:
    [[0, 0, 1], [0, 5, 0], [1, 1, 1]]
    Output Grid:
    [[0, 0, 0], [0, 2, 0], [1, 1, 1]]
    Reasoning: The '5' is replaced with a '2'. The rest of the grid is unchanged.

    Example 3:
    Input Grid:
    [[0, 0, 4], [0, 0, 4], [4, 4, 4]]
    Output Grid:
    [[0, 0, 4], [0, 0, 4], [4, 4, 4]]
    Reasoning: No transformation is apparent. The output is identical to the input.

    Now, apply the transformation you identified in the examples to the test input:
    Test Input:
    {question}

    Transformed Grid:
    """

    # Call the LLM to generate the transformed grid
    transformed_grid = call_llm(prompt)

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