import os
import re
import math

# HYPOTHESIS: Instead of trying to identify a single transformation rule, the LLM can directly transform the test grid 
# by considering the training examples as a set of constraints and desired outcomes. The key is to use a multi-example
# prompt that guides the LLM in applying transformations observed in the training data to the test grid *without*
# explicitly stating the rules. This leverages LLM's ability to generalize from examples.
# A validation loop is used to ensure the output is a valid grid.

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
        system_instruction = "You are an expert at transforming grids based on provided examples."
        prompt = f"""
        Given the following grid transformation problem, transform the test input grid according to the patterns observed in the training examples. Do NOT explicitly state the transformation rule. Apply the transformations directly.

        Example 1:
        === TRAINING EXAMPLES ===
        Input Grid:
        [[1, 2], [3, 4]]
        Output Grid:
        [[2, 3], [4, 5]]
        === TEST INPUT ===
        [[5, 6], [7, 8]]
        Transformed Grid:
        [[6, 7], [8, 9]]

        Example 2:
        === TRAINING EXAMPLES ===
        Input Grid:
        [[0, 1], [1, 0]]
        Output Grid:
        [[1, 0], [0, 1]]
        === TEST INPUT ===
        [[0, 0], [1, 1]]
        Transformed Grid:
        [[0, 0], [1, 1]]

        Problem:
        {question}

        Transformed Grid:
        """
        transformed_grid = call_llm(prompt, system_instruction)
        return transformed_grid

    # Function call is outside of the validation loop
    transformed_grid = transform_test_grid(question)

    def validate_grid_format(grid_string, max_attempts = 3):
      """Validates the output grid format as a list of lists."""
      system_instruction = "You are an expert grid validator. Your job is to validate the grid format as a list of lists."
      for attempt in range(max_attempts):
        validation_prompt = f"""
        Validate if the following grid string is a valid list of lists. The values in the lists may only be integer numbers.
          Grid String:
          {grid_string}

          The output must be a JSON object in the following format:
          {{
            "is_valid": true/false,
            "reason": "reasoning for output"
          }}
        """
        validation_json = call_llm(validation_prompt, system_instruction)

        if "true" in validation_json.lower():
          return True
        else:
          continue
      return False
    
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