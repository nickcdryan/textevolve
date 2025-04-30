import os
import re
import math

def main(question):
    """
    Solves grid transformation tasks by using a novel approach based on iterative 
    refinement using specialized LLM agents, with a focus on structured rule 
    representation and validation.

    Hypothesis: Combining structured rule representation with validation-driven 
    refinement improves the accuracy and robustness of grid transformations.
    """
    try:
        # 1. Extract data and represent it in a structured format
        extracted_data = extract_and_structure_data(question)
        if "Error" in extracted_data:
            return f"Data extraction error: {extracted_data}"

        # 2. Generate transformation rules in a structured format
        generated_rules = generate_transformation_rules(extracted_data)
        if "Error" in generated_rules:
            return f"Rule generation error: {generated_rules}"

        # 3. Refine transformation rules with validation
        refined_rules = refine_transformation_rules(extracted_data, generated_rules)
        if "Error" in refined_rules:
            return f"Rule refinement error: {refined_rules}"

        # 4. Apply refined transformation rules to test input
        transformed_grid = apply_transformation(extracted_data, refined_rules)
        if "Error" in transformed_grid:
            return f"Transformation application error: {transformed_grid}"

        return transformed_grid

    except Exception as e:
        return f"Unexpected error: {str(e)}"

def extract_and_structure_data(question):
    """
    Extracts relevant data and represents it in a structured JSON format,
    including error verification.
    """
    system_instruction = "You are an expert in extracting and structuring data from grid transformation problems."
    prompt = f"""
    Extract training examples and test input from the question. Represent the output as a JSON string.
    
    Example:
    Question: Grid Transformation Task
    Training Examples:
    [
        {{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}}
    ]
    Test Input: [[5, 6], [7, 8]]
    Extracted Data:
    {{
        "training_examples": [
            {{"input": "[[1, 2], [3, 4]]", "output": "[[4, 3], [2, 1]]"}}
        ],
        "test_input": "[[5, 6], [7, 8]]"
    }}
    
    Question: {question}
    Extracted Data:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error extracting data: {str(e)}"

def generate_transformation_rules(extracted_data):
    """
    Generates transformation rules in a structured, human-readable format using the extracted data.
    """
    system_instruction = "You are an expert at generating transformation rules for grid data."
    prompt = f"""
    Generate transformation rules from the training examples.
    Structure the rules in a human-readable format, focusing on patterns and relationships.

    Example:
    Extracted Data:
    {{
        "training_examples": [
            {{"input": "[[1, 2], [3, 4]]", "output": "[[4, 3], [2, 1]]"}}
        ],
        "test_input": "[[5, 6], [7, 8]]"
    }}
    Transformation Rules:
    "The grid is reflected along both diagonals. The element at position (i, j) is swapped with the element at position (N-1-i, N-1-j)."
    
    Extracted Data: {extracted_data}
    Transformation Rules:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error generating transformation rules: {str(e)}"

def refine_transformation_rules(extracted_data, generated_rules):
    """
    Refines the generated transformation rules by incorporating validation steps
    and handling edge cases.
    """
    system_instruction = "You are an expert at refining transformation rules to ensure accuracy."
    prompt = f"""
    Refine the generated transformation rules to ensure they are accurate and robust.
    Incorporate validation steps and consider edge cases.
    
    Example:
    Extracted Data:
    {{
        "training_examples": [
            {{"input": "[[1, 2], [3, 4]]", "output": "[[4, 3], [2, 1]]"}}
        ],
        "test_input": "[[5, 6], [7, 8]]"
    }}
    Generated Rules: "The grid is reflected along both diagonals."
    Refined Rules: "The grid is reflected along both diagonals. Validate that the grid is square before applying the transformation. If not, return an error. The element at position (i, j) is swapped with the element at position (N-1-i, N-1-j)."
    
    Extracted Data: {extracted_data}
    Generated Rules: {generated_rules}
    Refined Rules:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error refining transformation rules: {str(e)}"

def apply_transformation(extracted_data, refined_rules):
    """
    Applies the refined transformation rules to the test input and generates the transformed grid.
    """
    system_instruction = "You are an expert at applying transformation rules to grid data."
    prompt = f"""
    Apply the refined transformation rules to the test input. Generate the transformed grid based on the given rules.

    Example:
    Refined Rules: "The grid is reflected along both diagonals. The element at position (i, j) is swapped with the element at position (N-1-i, N-1-j)."
    Extracted Data:
    {{
        "training_examples": [
            {{"input": "[[1, 2], [3, 4]]", "output": "[[4, 3], [2, 1]]"}}
        ],
        "test_input": "[[5, 6], [7, 8]]"
    }}
    Transformed Grid: "[[8, 7], [6, 5]]"
    
    Refined Rules: {refined_rules}
    Extracted Data: {extracted_data}
    Transformed Grid:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error applying transformation: {str(e)}"

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