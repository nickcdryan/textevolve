import os
import re
import math

def main(question):
    """Transforms a grid based on patterns in training examples using a novel value propagation and contextual analysis approach."""
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem by first identifying value associations, then applying them contextually."""

    system_instruction = "You are an expert at identifying grid transformation patterns by analyzing value relationships and contextual dependencies. Focus on how values influence their neighbors and propagate across the grid."
    
    # STEP 1: Analyze value associations and relationships
    relationship_analysis_prompt = f"""
    You are tasked with identifying how values in a grid relate to each other during transformations.

    Example 1:
    Input Grid:
    [[1, 0], [0, 1]]
    Output Grid:
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    Analysis: The '1' values propagate diagonally across the expanding grid. Zeros fill the gaps.

    Example 2:
    Input Grid:
    [[2, 8], [8, 2]]
    Output Grid:
    [[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]
    Analysis: Each value replicates itself to form a 2x2 block.

    Example 3:
    Input Grid:
    [[0, 0, 0], [0, 0, 2], [2, 0, 2]]
    Output Grid:
    [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0, 2, 0, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0, 0, 2], [2, 0, 2, 0, 0, 0, 2, 0, 2]]
    Analysis: The value of '2' expands only to the bottom right of the initial position

    Now, analyze the value relationships and contextual dependencies in this example. Respond with ONLY the analysis:
    Test Example:
    {problem_text}
    """
    
    # Attempt to extract value relationships
    value_relationships = call_llm(relationship_analysis_prompt, system_instruction)

    # STEP 2: Infer transformation logic based on value relationships and test input
    transformation_logic_prompt = f"""
    Based on the identified value relationships:
    {value_relationships}

    And the test input grid:
    {problem_text}

    Infer the overall transformation logic. Consider how values interact, propagate, and modify the grid. Reason about the operations being performed. Respond with ONLY the transformation logic.
    """

    transformation_logic = call_llm(transformation_logic_prompt, system_instruction)
    
    # STEP 3: Apply the inferred transformation logic to the test input grid
    application_prompt = f"""
    You have extracted the following transformation logic:
    {transformation_logic}

    Now, apply this logic to the test input grid. Provide the transformed grid as a 2D array formatted as a string, WITHOUT any additional explanation or comments.
    Test Input Grid:
    {problem_text}
    """
    
    # Attempt to generate the transformed grid
    for attempt in range(max_attempts):
        try:
            transformed_grid_text = call_llm(application_prompt, system_instruction)
            # Basic validation - check if it looks like a grid
            if "[" in transformed_grid_text and "]" in transformed_grid_text:
                return transformed_grid_text
            else:
                print(f"Attempt {attempt+1} failed: Output does not resemble a grid. Retrying...")
        except Exception as e:
            print(f"Attempt {attempt+1} failed with error: {e}. Retrying...")

    # Fallback approach if all attempts fail
    return "[[0,0,0],[0,0,0],[0,0,0]]"

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