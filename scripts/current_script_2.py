import os
import re
import math

def solve_grid_transformation(question, max_attempts=3):
    """Solves grid transformation problems using localized contextual analysis.

    HYPOTHESIS: LLMs can identify grid transformation rules by focusing on local
    contexts within the grid rather than attempting to understand the global pattern
    all at once. The approach is to extract and analyze local contexts, and then
    synthesize the results to make predictions about the transformation of individual
    cells in the test input. It introduces targeted prompts for analyzing adjacent cell influences and validating patterns to avoid overfitting the limited training examples.

    APPROACH:
    1.  Identify key influencing factors by analyzing local cell contexts from the training examples.
    2.  Based on 1, predict the transformation of each cell in the test grid.

    """

    # Step 1: Analyze Local Contexts and Identify Influencing Factors
    context_analysis_result = analyze_local_contexts(question, max_attempts=max_attempts)
    if not context_analysis_result["is_valid"]:
        return f"Error: Could not identify influencing factors. {context_analysis_result['error']}"

    influencing_factors = context_analysis_result["influencing_factors"]

    # Step 2: Predict Cell Transformations based on Influencing Factors
    predicted_grid = predict_cell_transformations(question, influencing_factors, max_attempts=max_attempts)

    # Step 3: Verify and Refine Output Grid
    verification_result = verify_output_grid(question, predicted_grid, influencing_factors, max_attempts=max_attempts)
    if not verification_result["is_valid"]:
        return f"Error: Predicted grid validation failed. {verification_result['error']}"

    return predicted_grid

def analyze_local_contexts(question, max_attempts=3):
    """Analyzes local cell contexts to identify factors influencing transformations."""

    system_instruction = "You are an expert in analyzing grid transformations to identify local influences."

    for attempt in range(max_attempts):
        prompt = f"""
        Given the following grid transformation problem, analyze the local context of each cell to determine the key factors influencing its transformation.

        Example:
        Input Grid: [[1, 2, 1], [2, 1, 2], [1, 2, 1]]
        Output Grid: [[2, 3, 2], [3, 2, 3], [2, 3, 2]]
        Influencing Factors: The value of each cell in the output is the sum of itself and its immediate neighbors (up, down, left, right) in the input grid.

        Problem:
        {question}

        Influencing Factors:
        """

        influencing_factors = call_llm(prompt, system_instruction)

        # Validation step: check if the extracted factors are reasonable and non-contradictory
        validation_prompt = f"""
        Validate if the extracted influencing factors are reasonable and coherent.

        Problem: {question}
        Influencing Factors: {influencing_factors}

        Are these factors valid (True/False)?
        """

        is_valid = call_llm(validation_prompt, system_instruction)

        if "True" in is_valid:
            return {"is_valid": True, "influencing_factors": influencing_factors, "error": None}
        else:
            error_message = f"Invalid factors (attempt {attempt+1}): {influencing_factors}"
            print(error_message)
            if attempt == max_attempts - 1:
                return {"is_valid": False, "influencing_factors": None, "error": error_message}

    return {"is_valid": False, "influencing_factors": None, "error": "Failed to analyze local contexts."}

def predict_cell_transformations(question, influencing_factors, max_attempts=3):
    """Predicts the transformation of each cell based on the identified influencing factors."""

    system_instruction = "You are an expert at predicting grid cell transformations."

    for attempt in range(max_attempts):
        prompt = f"""
        Given the following grid transformation problem and the identified influencing factors, predict the transformation of each cell in the test input grid.

        Problem: {question}
        Influencing Factors: {influencing_factors}

        Test Input Grid: (extract from problem) ...

        Predicted Output Grid:
        """

        # Extract the test input grid from the problem description using regex
        test_input_match = re.search(r"=== TEST INPUT ===\n(\[.*?\])", question, re.DOTALL)
        if not test_input_match:
            return "Error: Could not extract test input grid."

        test_input_grid = test_input_match.group(1)

        # Construct a prediction prompt with the extracted test input
        prompt = f"""
        Given the following grid transformation problem and the identified influencing factors, predict the transformation of each cell in the test input grid.

        Problem: {question}
        Influencing Factors: {influencing_factors}
        Test Input Grid: {test_input_grid}

        Predicted Output Grid:
        """
        
        predicted_grid = call_llm(prompt, system_instruction)
        return predicted_grid

def verify_output_grid(question, output_grid, influencing_factors, max_attempts=3):
  for attempt in range(max_attempts):
        validation_prompt = f"""
        You are a meticulous grid transformation expert. 
        Problem: {question}
        Influencing Factors: {influencing_factors}
        Output Grid: {output_grid}

        1. Does the output grid follow the influencing factors?
        2. Is the output grid format correct and consistent with the examples in the problem?
        3. Is the output a valid Python list of lists representing the output grid?

        If there are issues, clearly explain what they are. If all checks pass, respond 'VALID'. Otherwise, explain the issues.
        """
        validation_result = call_llm(validation_prompt, system_instruction="You are a meticulous grid transformation expert.")

        if "VALID" in validation_result:
            return {"is_valid": True, "error": None}
        else:
            error_message = f"Validation failed (attempt {attempt + 1}): {validation_result}"
            print(error_message)
            if attempt == max_attempts - 1:
                return {"is_valid": False, "error": error_message}

  return {"is_valid": False, "error": "Failed verification after multiple attempts."}

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