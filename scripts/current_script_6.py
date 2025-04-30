import os
import re
import math

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

def analyze_transformation_pattern(training_examples):
    """Analyzes training examples to describe the transformation pattern."""
    system_instruction = "You are an expert in identifying transformation patterns in grid data."
    prompt = f"""
    Analyze the following training examples to describe the transformation pattern.
    
    Example 1:
    Input: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    Output: [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
    Transformation: The grid is reversed both horizontally and vertically.
    
    Example 2:
    Input: [[1, 2], [3, 4]]
    Output: [[1, 1], [3, 3]]
    Transformation: The second element of each row is replaced with the first element of that row.
    
    Example 3:
    Input: [[1, 2, 3], [4, 5, 6]]
    Output: [[2, 3, 4], [5, 6, 7]]
    Transformation: Each element is incremented by 1.
    
    Training Examples: {training_examples}
    Transformation:
    """
    return call_llm(prompt, system_instruction)

def apply_transformation(pattern_description, question):
    """Applies the transformation pattern to the test input."""
    system_instruction = "You are an expert in applying transformation patterns to grid data."
    prompt = f"""
    Apply the following transformation pattern to the test input.
    
    Example:
    Transformation: The grid is reversed both horizontally and vertically.
    Test Input: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    Transformed Grid: [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
    
    Transformation: The second element of each row is replaced with the first element of that row.
    Test Input: [[1, 2], [3, 4]]
    Transformed Grid: [[1, 1], [3, 3]]
    
    Transformation: Each element is incremented by 1.
    Test Input: [[1, 2, 3], [4, 5, 6]]
    Transformed Grid: [[2, 3, 4], [5, 6, 7]]
    
    Transformation: {pattern_description}
    Test Input: {question}
    Transformed Grid:
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to orchestrate the grid transformation process."""
    try:
        # Extract training examples and test input
        training_examples_str = re.search(r"Training Examples:\n(.*?)\nTest Input:", question, re.DOTALL)
        test_input_str = re.search(r"Test Input:\n(.*?)\nTransform", question, re.DOTALL)
        
        if not training_examples_str or not test_input_str:
            return "Error: Could not parse training examples or test input."
        
        training_examples = training_examples_str.group(1).strip()
        test_input = test_input_str.group(1).strip()
        
        # Analyze the transformation pattern
        pattern_description = analyze_transformation_pattern(training_examples)
        
        # Apply the transformation
        transformed_grid = apply_transformation(pattern_description, test_input)
        
        return transformed_grid
        
    except Exception as e:
        return f"Error: {str(e)}"