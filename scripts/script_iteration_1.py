import os
import re

def main(question):
    """
    Transforms a grid based on patterns identified in training examples, using LLM for reasoning.
    """
    try:
        # 1. Pattern Identification with Enhanced Examples and Targeted Improvements
        pattern_prompt = f"""
        You are an expert pattern recognition specialist for grid transformations.
        Identify the transformation pattern between the input and output grids in the following examples.
        Provide the pattern as a text description and also create code snippets to execute each transformation for each example.

        Example 1:
        Input Grid: [[1, 0, 0, 5, 0, 1, 0], [0, 1, 0, 5, 1, 1, 1], [1, 0, 0, 5, 0, 0, 0]]
        Output Grid: [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
        Pattern: The output grid is created by extracting the number of '1's to the left and right of the number 5 in the input grid. If the sum of those numbers equals 2, use that value as the number to place in the output grid. If the sum is anything else, use zero in the output grid. The output matrix is 3x3.
        Code:
        output = [[0, 0, 0] for _ in range(3)]
        input = [[1, 0, 0, 5, 0, 1, 0], [0, 1, 0, 5, 1, 1, 1], [1, 0, 0, 5, 0, 0, 0]]
        for i in range(len(input)):
            ones_left = 0
            ones_right = 0
            five_index = input[i].index(5)
            for j in range(five_index):
                if input[i][j] == 1:
                    ones_left += 1
            for j in range(five_index + 1, len(input[i])):
                if input[i][j] == 1:
                    ones_right += 1
            if (ones_left + ones_right) == 2:
                output[i][1] = 2

        Example 2:
        Input Grid: [[0, 1, 0, 0, 0, 0, 2], [1, 0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 2, 0, 0], [0, 0, 0, 2, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 4], [2, 0, 0, 0, 0, 4, 0]]
        Output Grid: [[2, 1, 4, 2, 1, 4, 2], [1, 4, 2, 1, 4, 2, 1], [4, 2, 1, 4, 2, 1, 4], [2, 1, 4, 2, 1, 4, 2], [1, 4, 2, 1, 4, 2, 1], [4, 2, 1, 4, 2, 1, 4], [2, 1, 4, 2, 1, 4, 2]]
        Pattern: The output grid is created from replicating the numbers in the diagonals from the original grid.
        Code:
        def solve():
            input_grid = [[0, 1, 0, 0, 0, 0, 2], [1, 0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 2, 0, 0], [0, 0, 0, 2, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 4], [2, 0, 0, 0, 0, 4, 0]]
            n = len(input_grid)
            output_grid = [[0] * n for _ in range(n)]

            for i in range(n):
                for j in range(n):
                    if input_grid[i][j] != 0:
                        for row in range(n):
                            output_grid[row][(row + j - i + n) % n] = input_grid[i][j]
            print(output_grid)

        Now, identify the pattern for the following input:
        {question}
        """

        pattern_response = call_llm(pattern_prompt, "You are a pattern recognition expert.")

        # 2. Transformation Plan - Create an actionable plan.
        plan_prompt = f"""
        Based on your identified transformation pattern:
        {pattern_response}

        Generate a detailed step-by-step plan to transform the input grid.
        Include python code snippets to accomplish each step.  Assume there is an input grid named 'input_grid'.

        Example plan:
        1. Create an empty output grid with dimensions matching the input grid.
        2. Iterate through each cell in the input grid.
        3. If the cell value is greater than 5, set the corresponding cell in the output grid to 1.
        4. Otherwise, set the output grid cell to 0.

        Output the plan as a numbered list, with a code block to execute the identified plan with input 'input_grid', creating an output 'output_grid'.
        """
        plan_response = call_llm(plan_prompt, "You are a transformation planning expert.")

        # 3. Execution of the Transformation Plan - Execute generated code.
        execution_prompt = f"""
        You have a plan to transform an input grid named 'input_grid' and to output a transformed grid called 'output_grid':
        {plan_response}

        Given the following test input grid, execute the plan directly in Python code, and return ONLY the final 'output_grid':
        {question}

        Your response should include a code block that defines input_grid from the provided question, and then executes the transformations according to the provided plan. The result of the last line of code should define the 'output_grid' variable, and no other output or information is necessary.

        Example:
        input_grid = [[1, 0, 0, 5, 0, 1, 0], [0, 1, 0, 5, 1, 1, 1], [1, 0, 0, 5, 0, 0, 0]]
        output_grid = [[0, 0, 0] for _ in range(3)]
        for i in range(len(input_grid)):
            ones_left = 0
            ones_right = 0
            five_index = input_grid[i].index(5)
            for j in range(five_index):
                if input_grid[i][j] == 1:
                    ones_left += 1
            for j in range(five_index + 1, len(input_grid[i])):
                if input_grid[i][j] == 1:
                    ones_right += 1
            if (ones_left + ones_right) == 2:
                output_grid[i][1] = 2
        print(output_grid)
        """

        execution_code = call_llm(execution_prompt, "You are an expert Python code executor.")

        # Safely execute the generated code.
        local_vars = {}
        exec(execution_code, globals(), local_vars)
        output_grid = local_vars.get('output_grid', "Error: output_grid not defined")

        return str(output_grid) # Ensuring we return a string
    except Exception as e:
        return f"Error: {str(e)}"

# Helper function for calling the LLM - DO NOT MODIFY
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