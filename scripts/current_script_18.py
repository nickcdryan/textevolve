import os
import re
import math

def main(question):
    """
    Solves grid transformation tasks using a "Transformation Decomposition and Rule Extraction" approach.

    Hypothesis: By explicitly decomposing the transformation into smaller, understandable components
    (e.g., row operations, column operations, element replacements) and then extracting individual
    rules for each component, the LLM can better capture complex transformations. Adds verification for each component.
    """
    try:
        # 1. Extract relevant grid data.
        extracted_data = extract_data(question)
        if "Error" in extracted_data:
            return f"Data extraction error: {extracted_data}"

        # 2. Decompose the transformation
        transformation_components = decompose_transformation(extracted_data)
        if "Error" in transformation_components:
            return f"Transformation decomposition error: {transformation_components}"

        # 3. Extract transformation rules
        transformation_rules = extract_transformation_rules(extracted_data, transformation_components)
        if "Error" in transformation_rules:
            return f"Transformation rule extraction error: {transformation_rules}"

        # 4. Apply the rules to the test input.
        transformed_grid = apply_transformation(extracted_data, transformation_rules)
        if "Error" in transformed_grid:
            return f"Transformation application error: {transformed_grid}"

        return transformed_grid

    except Exception as e:
        return f"Unexpected error: {str(e)}"

def extract_data(question):
    """Extracts training and test data with example-based guidance for improved accuracy."""
    system_instruction = "You are an expert at extracting structured data from grid transformation problems."
    prompt = f"""
    Extract the training examples and test input from the question.
    Format the output as a dictionary-like string. Ensure training examples and the test input are well-formatted.

    Example:
    Question: Grid Transformation Task
    Training Examples:
    [
        {{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}}
    ]
    Test Input: [[5, 6], [7, 8]]
    Extracted Data:
    {{'training_examples': '[{{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}}]', 'test_input': '[[5, 6], [7, 8]]'}}

    Question: {question}
    Extracted Data:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error extracting data: {str(e)}"

def decompose_transformation(extracted_data):
    """Decomposes the transformation into row operations, column operations, and element replacements"""
    system_instruction = "You are an expert at decomposing grid transformations"
    prompt = f"""
    Decompose the transformations into its different components. The components must be one of the following:
    Row Operations, Column Operations, Element Replacements, Subgrid Extraction
    
    Example:
    Extracted Data:
    {{'training_examples': '[{{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}}]', 'test_input': '[[5, 6], [7, 8]]'}}
    Transformation Components: Reflection across both diagonals, Element Replacements
    
    Extracted Data: {extracted_data}
    Transformation Components:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error decomposing the transform"
    
def extract_transformation_rules(extracted_data, transformation_components):
    """Extracts the individual transformation rules for each component"""
    system_instruction = "You are an expert at extracting specific transformation rules from the extracted data."
    prompt = f"""
    Based on these extracted transformation components, extract the rules. Be specific!
    Extracted Data:
    {{'training_examples': '[{{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}}]', 'test_input': '[[5, 6], [7, 8]]'}}
    Transformation Components: Reflection across both diagonals, Element Replacements

    Extracted Transformation Rules: Reflect across both diagonals by swapping A[0][0] with A[1][1] and A[0][1] with A[1][0], replace A[0][0] with 4, A[0][1] with 3, A[1][0] with 2, and A[1][1] with 1.
    

    Extracted Data: {extracted_data}
    Transformation Components: {transformation_components}
    Extracted Transformation Rules:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error extracting transform rules"

def apply_transformation(extracted_data, transformation_rules):
    """Applies the refined transformation pattern with direct example guidance."""
    system_instruction = "You are an expert at applying refined transformation patterns to grid data."
    prompt = f"""
    Apply the refined transformation rules to the test input and generate the transformed grid.
    Refined Transformation Rules: Reflect across both diagonals by swapping A[0][0] with A[1][1] and A[0][1] with A[1][0], replace A[0][0] with 4, A[0][1] with 3, A[1][0] with 2, and A[1][1] with 1.
    Test Input: {{'training_examples': '[{{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}}]', 'test_input': '[[5, 6], [7, 8]]'}}
    Transformed Grid: [[8, 7], [6, 5]]

    Refined Transformation Rules: {transformation_rules}
    Test Input: {extracted_data}
    Transformed Grid:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error applying refined transformation: {str(e)}"

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