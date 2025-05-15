import os
import re
import math

def solve_grid_transformation(question, max_attempts=3):
    """Solves grid transformation problems by detecting and applying transformations to individual elements.
    HYPOTHESIS: The core transformation can be extracted by focusing on the individual elements within the grid
    rather than trying to understand the whole grid at once. We break down the question into distinct steps:
    1. Find the unique elements within the grid and how they get transformed
    2. Identify the transformation rules that transform those elements
    3. Apply the transformation rules to generate new grids

    APPROACH:
    1. Find unique elements in input and output grids.
    2. Infer transformations for each unique element.
    3. Apply these transformations to the test input grid.
    4. Verification of output to attempt to resolve any potential problems with processing
    """

    # Step 1: Extract unique elements and their transformations
    element_analysis_result = analyze_elements(question, max_attempts=max_attempts)
    if not element_analysis_result["is_valid"]:
        return f"Error: Could not analyze elements. {element_analysis_result['error']}"

    element_transformations = element_analysis_result["element_transformations"]

    # Step 2: Apply element transformations to the test input
    predicted_grid = transform_grid(question, element_transformations, max_attempts=max_attempts)

    # Step 3: Verfiy output grid
    verification_result = verify_output_grid(question, predicted_grid, element_transformations, max_attempts=max_attempts)

    if not verification_result["is_valid"]:
        return f"Error: verification failure {verification_result['error']}"

    return predicted_grid

def analyze_elements(question, max_attempts=3):
    """Analyzes input and output grids to find unique elements and their transformations."""

    system_instruction = "You are an expert at analyzing grid transformations to identify element-level transformations."

    for attempt in range(max_attempts):
        prompt = f"""
        Given the following grid transformation problem, identify the unique elements in the input and output grids and determine how they are transformed.

        Example:
        Input Grid: [[1, 2, 1], [2, 1, 2], [1, 2, 1]]
        Output Grid: [[2, 3, 2], [3, 2, 3], [2, 3, 2]]
        Element Transformations:
        1 -> 2
        2 -> 3

        Problem:
        {question}

        Element Transformations:
        """

        element_transformations = call_llm(prompt, system_instruction)

        # Validation: check if the extracted transformations are reasonable
        validation_prompt = f"""
        Validate the extracted element transformations for the given problem.

        Problem: {question}
        Extracted Transformations: {element_transformations}

        Are these transformations valid (True/False)?
        """

        is_valid = call_llm(validation_prompt, system_instruction)

        if "True" in is_valid:
            return {"is_valid": True, "element_transformations": element_transformations, "error": None}
        else:
            error_message = f"Invalid transformations (attempt {attempt+1}): {element_transformations}"
            print(error_message)
            if attempt == max_attempts - 1:
                return {"is_valid": False, "element_transformations": None, "error": error_message}

    return {"is_valid": False, "element_transformations": None, "error": "Failed to analyze elements."}

def transform_grid(question, element_transformations, max_attempts=3):
    """Applies element transformations to the test input grid."""
    system_instruction = "You are an expert at applying element transformations to grids."
    for attempt in range(max_attempts):
        # Extract the test input grid from the problem description using regex
        test_input_match = re.search(r"=== TEST INPUT ===\n(\[.*?\])", question, re.DOTALL)
        if not test_input_match:
            return "Error: Could not extract test input grid."

        test_input_grid = test_input_match.group(1)
        prompt = f"""
        Given the following grid transformation problem and element transformations, apply the transformations to the test input grid to generate the output grid.

        Problem: {question}
        Element Transformations: {element_transformations}
        Test Input Grid: {test_input_grid}

        Output Grid:
        """
        predicted_grid = call_llm(prompt, system_instruction)

        #attempt verification at each generation
        validation_prompt = f"""
        Given the transformation problem, and the predicted grid, determine if the transformations are correct. Return "VALID" if the solution looks correct and "INVALID" if it looks incorrect.
        If incorrect provide an explanation as to why.
        Problem: {question}
        Output Grid: {predicted_grid}
        """

        is_valid = call_llm(validation_prompt, system_instruction)
        if "VALID" in is_valid:
            return predicted_grid
        else:
            error_message = f"Grid transformations invalid (attempt {attempt+1}): {predicted_grid}"
            print(error_message)
            if attempt == max_attempts - 1:
                return f"Could not transform grid"

        return predicted_grid
def verify_output_grid(question, output_grid, element_transformations, max_attempts=3):
    """Verifies the output grid against the transformation rules."""
    system_instruction = "You are a grid transformation expert, responsible for verifying output grids."

    for attempt in range(max_attempts):
        validation_prompt = f"""
            Verify the output grid against the problem statement and transformations

            Problem: {question}
            Element Transformations: {element_transformations}
            Output Grid: {output_grid}

            Does the output match transformation requirements? (VALID/INVALID)
            """
        validation_result = call_llm(validation_prompt, system_instruction)
        if "VALID" in validation_result:
            return {"is_valid": True, "error": None}
        else:
            return {"is_valid": False, "error": "The result is not valid"}

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

def main(question):
    """Main function to solve the grid transformation task."""
    try:
        answer = solve_grid_transformation(question)
        return answer
    except Exception as e:
        return f"Error in main function: {str(e)}"