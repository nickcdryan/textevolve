import os
import re
import math

# Hypothesis: This exploration will focus on simplifying the approach and using a direct, example-based LLM transformation with a pre-processing and post-processing step to fix format.
# We hypothesize that a direct approach with strong formatting instructions and post-formatting correction will improve reliability.

def main(question):
    """Transforms a grid based on examples, using LLM for direct transformation with format correction."""
    try:
        # 1. Preprocess the question - extract input and training examples
        preprocessed_data = preprocess_question(question)

        # 2. Direct Grid Transformation with strong formatting.
        transformed_grid = transform_grid_with_formatting(preprocessed_data)

        # 3. Attempt Post-Processing fix if needed
        final_grid = post_process_grid(transformed_grid)

        return final_grid
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def preprocess_question(question):
    """Extracts the input grid and training examples from the question."""
    # Simple regex to extract training examples and test input.
    training_examples_match = re.search(r"=== TRAINING EXAMPLES ===\n(.*?)\n=== TEST INPUT ===", question, re.DOTALL)
    test_input_match = re.search(r"=== TEST INPUT ===\n(.*?)\nTransform", question, re.DOTALL)

    training_examples = training_examples_match.group(1).strip() if training_examples_match else ""
    test_input = test_input_match.group(1).strip() if test_input_match else ""

    return {"training_examples": training_examples, "test_input": test_input}

def transform_grid_with_formatting(data):
    """Transforms the grid using direct LLM call with formatting instructions."""
    system_instruction = "You are an expert grid transformer. You MUST ALWAYS output a valid grid string that starts with '[[' and ends with ']]'."
    prompt = f"""
    You are a grid transformation expert. Analyze the training examples and transform the test input accordingly.
    Your response MUST be ONLY the transformed grid, a string representation that starts with '[[' and ends with ']]'.
    Do NOT include any additional explanations or reasoning steps in your answer.
    The numbers MUST be comma separated.
    
    Training Examples:
    {data['training_examples']}

    Test Input:
    {data['test_input']}
    

    Here's an example input/output pair with a corresponding reasoning.
    Input: [[1, 2], [3, 4]]
    Reasoning: This appears to just add 1 to each number.
    Output: [[2, 3], [4, 5]]
    
    Another example of correct formatting of the output with no additional information:
    Input: [[1, 2], [3, 4]]
    Reasoning: This appears to duplicate each number.
    Output: [[1, 1, 2, 2], [3, 3, 4, 4]]

    Now transform the test input. Your output *MUST* start with '[[' and end with ']]' and only contain the grid string:
    Transformed Grid:
    """

    transformed_grid = call_llm(prompt, system_instruction)
    return transformed_grid #No validation at this step to let post processing do its work

def post_process_grid(grid_string):
    """Attempts to fix formatting errors in the grid string."""
    # Remove extra text before '[[' and after ']]'
    try:
        start_index = grid_string.find("[[")
        end_index = grid_string.rfind("]]")

        if start_index == -1 or end_index == -1:
            return "ERROR: Could not find '[[' or ']]' in the output."
        
        cleaned_grid = grid_string[start_index:end_index+2]

        #Attempt minimal cleaning of extra spaces. We are explicitly AVOIDING json.loads() here!
        cleaned_grid = cleaned_grid.replace(" ", "")

        # Return the cleaned grid - if its a problem, then its a later stage problem
        return cleaned_grid

    except Exception as e:
        return f"ERROR in post-processing: {str(e)}"
        
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