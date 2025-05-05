import os
import re
import math

# Hypothesis: This exploration will focus on a new approach by using the LLM to do a "similarity search" between 
# training and test inputs to identify the most relevant example, and then use that example to guide the transformation.
# We are essentially trying to teach the LLM to select its best few-shot example on its own based on similarity.
# We hypothesize that focusing on the most relevant example will produce better results than using all examples or random examples.

def main(question):
    """Transforms a grid based on identifying the most similar training example and applying its transformation."""
    try:
        # 1. Identify the most similar example using LLM
        most_similar_example = identify_most_similar_example(question)

        # 2. Apply the transformation based on the identified example
        transformed_grid = apply_transformation_from_example(question, most_similar_example)

        return transformed_grid
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def identify_most_similar_example(question):
    """Identifies the most similar training example using LLM."""
    system_instruction = "You are an expert in identifying similarities between grid transformation examples."
    prompt = f"""
    You are an expert in identifying similarities between grid transformation examples.
    Given a question containing training examples and a test input, identify the training example that is most similar to the test input.
    Return ONLY the content of the most similar example, WITHOUT any additional text or explanations.
    
    Here's how you should reason through the problem. First, extract the training examples and test input.
    Second, compare the test input with each training example in turn to evaluate similarity
    Third, explain WHY a training example has the MOST SIMILAR qualities to the test input - for example the same number of rows and cols or some other specific element. 

    Example:
    Question:
    Grid Transformation Task

    === TRAINING EXAMPLES ===

    Example 1:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[2, 3], [4, 5]]

    Example 2:
    Input Grid: [[1, 0], [0, 1]]
    Output Grid: [[2, 1], [1, 2]]

    === TEST INPUT ===
    [[5, 6], [7, 8]]
    
    Reasoning: This test input has similar dimensions to training example 1, 2x2.
    The numbers also appear to increment in order within each row, as is the case in example 1

    Most Similar Example:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[2, 3], [4, 5]]
    
    Here's another, different, example:
    Question:
    Grid Transformation Task

    === TRAINING EXAMPLES ===

    Example 1:
    Input Grid: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    Output Grid: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    Example 2:
    Input Grid: [[0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    Output Grid: [[0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    === TEST INPUT ===
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 7, 2, 7, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 7, 2, 7, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    
    Reasoning: This test input has more rows and different col counts than other examples. Furthermore it seems to contain some basic '7' to '2' transformations. Even though test input has more rows, example 2, with its longer row than example 1, is most similar, and therefore should be used. It also contains a '2'.

    Most Similar Example:
    Input Grid: [[0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    Output Grid: [[0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    Now, for this new question, identify the most similar example:
    {question}
    """
    response = call_llm(prompt, system_instruction)
    return response.strip()

def apply_transformation_from_example(question, example):
    """Applies the transformation from the most similar example to the test input."""
    system_instruction = "You are an expert in applying grid transformations based on examples."
    prompt = f"""
    You are an expert in applying grid transformations based on examples.
    Given a question containing a test input and the most similar training example, apply the transformation from the example to the test input.
    Return ONLY the transformed grid, WITHOUT any additional text or explanations.

    Question:
    {question}

    Most Similar Example:
    {example}

    Reasoning: Apply a similar transformation to the test input as demonstrated by the example.

    Example of correct formatting of the output with no additional information:
    Input: [[1, 2], [3, 4]]
    Reasoning: This appears to just add 1 to each number.
    Output: [[2, 3], [4, 5]]
    
    Now transform the test input. Your output *MUST* start with '[[' and end with ']]' and only contain the grid string:
    Transformed Grid:
    """
    transformed_grid = call_llm(prompt, system_instruction)
    return transformed_grid #No validation at this step to let post processing do its work
        
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