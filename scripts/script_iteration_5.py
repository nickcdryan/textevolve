import os
import re
import math

def solve_grid_transformation(question, max_attempts=3):
    """Solves grid transformation problems by analyzing row and column patterns independently.
    
    HYPOTHESIS: By decomposing the problem into independent row and column analyses, we can simplify the pattern recognition process and improve the LLM's ability to identify the underlying transformation rules. This is based on the observation that some grid transformations operate primarily on rows or columns. If neither of these analyses provide good answers, a fallback to a whole grid analysis is performed.

    APPROACH:
    1. Analyze row transformations.
    2. Analyze column transformations.
    3. Attempt to recombine the row and column transformations.
    4. If both the row/column transformations fail, fall back to a whole grid analysis.
    """

    row_analysis_result = analyze_row_transformations(question, max_attempts=max_attempts)
    column_analysis_result = analyze_column_transformations(question, max_attempts=max_attempts)
    
    if row_analysis_result["is_valid"] and column_analysis_result["is_valid"]:
        #Attempt to combine the answers from the rows and columns together.
        combined_result = combine_row_column_results(question, row_analysis_result["transformed_grid"], column_analysis_result["transformed_grid"])
        return combined_result
    else:
        whole_grid_result = analyze_whole_grid(question, max_attempts=max_attempts)
        return whole_grid_result["transformed_grid"]

def analyze_row_transformations(question, max_attempts=3):
    """Analyzes the input and output grids to identify row-based transformations."""
    system_instruction = "You are an expert at identifying row-based transformations in grids."
    for attempt in range(max_attempts):
        prompt = f"""
        Given the following grid transformation problem, analyze the training examples to identify any patterns in how the rows are transformed from the input grid to the output grid.

        Example:
        Question:
        === TRAINING EXAMPLES ===
        Input Grid:
        [[1, 2, 3], [4, 5, 6]]
        Output Grid:
        [[2, 3, 4], [5, 6, 7]]
        Row Transformation: Each number in the row is incremented by 1.

        Problem:
        {question}

        Row Transformation:
        """
        row_transformation = call_llm(prompt, system_instruction)

        validation_prompt = f"""
        Validate if the extracted row transformation is correct and can be applied to all rows in the training examples.
        Problem: {question}
        Row Transformation: {row_transformation}
        Is the row transformation valid? (True/False)
        """
        is_valid = call_llm(validation_prompt, system_instruction)

        if "True" in is_valid:
            transformed_grid = apply_row_transformation(question, row_transformation)
            return {"is_valid": True, "transformed_grid": transformed_grid, "error": None}
        else:
            error_message = f"Invalid row transformation (attempt {attempt+1}): {row_transformation}"
            print(error_message)
            if attempt == max_attempts - 1:
                return {"is_valid": False, "transformed_grid": None, "error": error_message}
    return {"is_valid": False, "transformed_grid": None, "error": "Failed to analyze row transformations."}

def analyze_column_transformations(question, max_attempts=3):
    """Analyzes the input and output grids to identify column-based transformations."""
    system_instruction = "You are an expert at identifying column-based transformations in grids."
    for attempt in range(max_attempts):
        prompt = f"""
        Given the following grid transformation problem, analyze the training examples to identify any patterns in how the columns are transformed from the input grid to the output grid.

        Example:
        Question:
        === TRAINING EXAMPLES ===
        Input Grid:
        [[1, 4], [2, 5], [3, 6]]
        Output Grid:
        [[2, 5], [3, 6], [4, 7]]
        Column Transformation: Each number in the column is incremented by 1.

        Problem:
        {question}

        Column Transformation:
        """
        column_transformation = call_llm(prompt, system_instruction)

        validation_prompt = f"""
        Validate if the extracted column transformation is correct and can be applied to all columns in the training examples.
        Problem: {question}
        Column Transformation: {column_transformation}
        Is the column transformation valid? (True/False)
        """
        is_valid = call_llm(validation_prompt, system_instruction)

        if "True" in is_valid:
            transformed_grid = apply_column_transformation(question, column_transformation)
            return {"is_valid": True, "transformed_grid": transformed_grid, "error": None}
        else:
            error_message = f"Invalid column transformation (attempt {attempt+1}): {column_transformation}"
            print(error_message)
            if attempt == max_attempts - 1:
                return {"is_valid": False, "transformed_grid": None, "error": error_message}
    return {"is_valid": False, "transformed_grid": None, "error": "Failed to analyze column transformations."}

def analyze_whole_grid(question, max_attempts=3):
  system_instruction = "You are an expert at transforming grids and solving grid transformation problems."
  for attempt in range(max_attempts):
      prompt = f"""
      Given the following grid transformation problem, analyze the whole grid and apply a transformation in order to solve the grid transformation problem.

      Example:
      Question:
      === TRAINING EXAMPLES ===
      Input Grid:
      [[1, 2], [3, 4]]
      Output Grid:
      [[2, 3], [4, 5]]
      Transformation: Each number in the grid is incremented by 1.

      Problem:
      {question}

      Transformation:
      """
      transformation = call_llm(prompt, system_instruction)

      validation_prompt = f"""
        Validate the given grid transformation in order to solve the given grid transformation problem.
        Problem: {question}
        Output Grid: {transformation}
        Is the solution valid? (True/False)
        """
      is_valid = call_llm(validation_prompt, system_instruction)

      if "True" in is_valid:
          return {"is_valid": True, "transformed_grid": transformation, "error": None}
      else:
          error_message = f"Invalid grid transformation (attempt {attempt+1}): {transformation}"
          print(error_message)
          if attempt == max_attempts - 1:
              return {"is_valid": False, "transformed_grid": None, "error": error_message}

  return {"is_valid": False, "transformed_grid": None, "error": "Failed to analyze grid transformations."}

def combine_row_column_results(question, row_transformed_grid, column_transformed_grid):
  """Combines results and takes the transformed grid from the approach that validates as true."""
  system_instruction = "You are an expert grid solution result combiner. You will review proposed results from different grid approaches and select the result that looks the best and most promising."

  prompt = f"""
        You have proposed solution grids based on different transformation approaches, and your job is to select the best solution, given the original problem.
        The better and more correct solution will be chosen and given as the result. The problem is given, along with the proposed row and column transformation results.
        If one of the two results is a clear and well-formed grid result, and the other is not well-formed, or if the quality of one output looks better, then that should be chosen.

        Problem: {question}
        Row Transformation Result: {row_transformed_grid}
        Column Transformation Result: {column_transformed_grid}
        Result:
        """
  result = call_llm(prompt, system_instruction)
  return result

def apply_row_transformation(question, row_transformation):
    """Applies the extracted row transformation to the test input grid."""
    system_instruction = "You are an expert at applying row transformations to grids."
    prompt = f"""
    Given the following grid transformation problem and the extracted row transformation, apply the row transformation to the test input grid.

    Problem: {question}
    Row Transformation: {row_transformation}

    Generate the output grid by applying the row transformation.
    """
    output_grid = call_llm(prompt, system_instruction)
    return output_grid

def apply_column_transformation(question, column_transformation):
    """Applies the extracted column transformation to the test input grid."""
    system_instruction = "You are an expert at applying column transformations to grids."
    prompt = f"""
    Given the following grid transformation problem and the extracted column transformation, apply the column transformation to the test input grid.

    Problem: {question}
    Column Transformation: {column_transformation}

    Generate the output grid by applying the column transformation.
    """
    output_grid = call_llm(prompt, system_instruction)
    return output_grid

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