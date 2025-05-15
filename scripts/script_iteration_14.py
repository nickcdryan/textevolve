import os
import re
import math

# HYPOTHESIS: Instead of analyzing visual features, the LLM can be used to directly generate the output grid
# by learning a transformation function represented implicitly in the examples. This relies on LLM's powerful
# few-shot learning abilities. A validation loop is used to make sure the output is a list of lists and contains numbers

def solve_grid_transformation(question, max_attempts=3):
    """Solves grid transformation problems by direct example-guided transformation."""

    def call_llm(prompt, system_instruction=None):
        """Call the Gemini LLM with a prompt and return the response."""
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

    def transform_test_grid(question, max_attempts=3):
        """Transforms the test grid based on training examples."""
        system_instruction = "You are an expert at transforming grids based on provided examples. You transform the grids in the same list of list format."
        prompt = f"""
        Given the following grid transformation problem, transform the test input grid according to the patterns observed in the training examples. You are to produce the output grid in the same format, as a list of lists of numbers.
        
        Example 1:
        Problem:
        === TRAINING EXAMPLES ===
        Input Grid:
        [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        Output Grid:
        [[1, 1, 1], [0, 0, 0], [1, 1, 1]]
        === TEST INPUT ===
        [[0, 0, 0], [2, 2, 2], [0, 0, 0]]
        Transformed Grid:
        [[2, 2, 2], [0, 0, 0], [2, 2, 2]]

        Example 2:
        Problem:
        === TRAINING EXAMPLES ===
        Input Grid:
        [[1, 0], [0, 1]]
        Output Grid:
        [[0, 1], [1, 0]]
        === TEST INPUT ===
        [[0, 0], [1, 1]]
        Transformed Grid:
        [[1, 0], [0, 0]]

        Problem:
        {question}

        Transformed Grid:
        """
        transformed_grid = call_llm(prompt, system_instruction)
        return transformed_grid

    def validate_grid_format(grid_string, max_attempts = 3):
        """Validates the output grid format as a list of lists."""
        system_instruction = "You are an expert grid validator. Your job is to validate the grid format. Respond with VALID or INVALID."
        for attempt in range(max_attempts):
          validation_prompt = f"""
          Validate if the following string is a valid list of lists and has only numbers.
            String:
            {grid_string}

            Respond with VALID if the grid has the correct format, and respond with INVALID if not.
          """
          validation_json = call_llm(validation_prompt, system_instruction)

          if "VALID" in validation_json.upper():
            return True
          else:
            continue
        return False
    
    transformed_grid = transform_test_grid(question)
    
    if validate_grid_format(transformed_grid):
      return transformed_grid
    else:
      return "Error: Could not create a valid transformation"

def main(question):
    """Main function to solve the grid transformation task."""
    try:
        answer = solve_grid_transformation(question)
        return answer
    except Exception as e:
        return f"Error in main function: {str(e)}"