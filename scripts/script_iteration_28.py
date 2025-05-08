import os
import re
import math

def main(question):
    """Transforms a grid based on patterns in training examples using LLM-driven pattern recognition.
    This approach leverages a "Transformation Decomposition and Value Prediction" strategy.
    Hypothesis: Decomposing the transformation into smaller, predictable steps and focusing on value prediction will improve accuracy.
    """
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem by decomposing and predicting transformations."""

    system_instruction = "You are an expert at identifying grid transformation patterns and applying them. Decompose the transformation and predict value changes."
    
    # STEP 1: Decompose the Transformation into steps - with examples!
    decomposition_prompt = f"""
    Decompose the grid transformation into a series of steps. Identify:
    1. What aspects of the grid are changing (e.g., size, shape, values)?
    2. What triggers each change (e.g., location, neighboring values)?
    3. What are the individual operations to perform each change?

    Example 1:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[1, 0], [0, 1]]\n\nOutput Grid:\n[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]\n
    Decomposition:
    1. Grid size is increasing.
    2. Original values are placed on the diagonal.
    3. Non-diagonal values are filled with 0.

    Example 2:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[2, 8], [8, 2]]\n\nOutput Grid:\n[[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]\n
    Decomposition:
    1. Each element is expanding.
    2. Each element becomes a 2x2 block.
    3. The 2x2 block has the same value as the original element.

    Problem: {problem_text}
    Decomposition:
    """
    
    # Attempt to decompose the transformation
    extracted_decomposition = call_llm(decomposition_prompt, system_instruction)
    print(f"Extracted Decomposition: {extracted_decomposition}") # Diagnostic

    # STEP 2: Value Prediction - predict the value of a cell at a specific location. WITH EXAMPLES!
    value_prediction_prompt = f"""
    Given the transformation decomposition, predict the value of the cell at a specific location in the output grid. Show step-by-step reasoning.
    Decomposition: {extracted_decomposition}
    Test Input Grid: {problem_text}

    Example:
    Decomposition:
    1. Grid size is increasing.
    2. Original values are placed on the diagonal.
    3. Non-diagonal values are filled with 0.
    Input Grid: [[1, 0], [0, 1]]
    Location: (0, 0)
    Reasoning: (0,0) is a diagonal, therefore it should keep the original value from the source grid = 1.
    Predicted Value: 1

    Location: (0, 1)
    Reasoning: (0,1) is NOT on the diagonal, therefore the value should be zero.
    Predicted Value: 0

    New Input:
    Test Input Grid: [[2, 8], [8, 2]]
    Location: (0, 0)
    Reasoning: Each element is expanding to a 2x2 block. At (0,0), the value should be the same as the top left. Top left is 2, therefore output should be 2.
    Predicted Value: 2

    What is the predicted value for location (0, 0)? Explain your reasoning.
    """
    
    #Attempt to predict the value
    predicted_value = call_llm(value_prediction_prompt, system_instruction)
    print(f"Predicted Value: {predicted_value}")

    #STEP 3: Construct the Transformed Grid from Predicted Values (Rudimentary attempt)
    try:
       value = re.search(r"(\d+)", predicted_value).group(1) #extracts the numerical predicted value
       transformed_grid_text = f"[[{value}]]"  #basic grid with one value
    except:
       transformed_grid_text = "[[0]]"
    
    #Basic validation to ensure output is not empty
    if not transformed_grid_text:
        transformed_grid_text = "[[0]]" #Fallback if extraction failed

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