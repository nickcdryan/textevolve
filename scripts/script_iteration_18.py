import os
import re
import math

# Hypothesis: This exploration will implement a "Transformation by Iterative Local Contextual Adjustment with Explicit Constraints" approach.
# This is a new hybrid approach. Instead of trying to get the LLM to understand complex patterns directly, we'll focus on local, iterative adjustments.
# 1. LLM analyzes local context of EACH CELL in the grid.
# 2. LLM proposes a *small* adjustment to that cell value.
# 3. A Constraint Checker module DETERMINISTICALLY enforces explicit constraints (e.g., total sum of row must be constant).
# The key here is that we're not asking the LLM to solve the whole problem at once, but to make *incremental* improvements guided by constraints.
# Add verification steps to understand which parts are successful and where it is breaking.

def main(question):
    """Transforms a grid by iteratively adjusting cell values within constraints."""
    try:
        # 1. Extract training examples and test input
        training_examples, test_input = preprocess_question(question)

        # 2. Initialize the grid
        grid = initialize_grid(test_input)

        # 3. Iteratively adjust cell values
        for _ in range(3):  # Number of iterations
            for row in range(len(grid)):
                for col in range(len(grid[0])):
                    grid = adjust_cell_value(grid, row, col, training_examples)
                    grid = enforce_constraints(grid) # Apply constraints after each adjustment
        
        #4. return results
        return str(grid)

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def preprocess_question(question):
    """Extract training examples and test input from the question string using regex."""
    try:
        training_examples_match = re.search(r"=== TRAINING EXAMPLES ===\n(.*?)\n=== TEST INPUT ===", question, re.DOTALL)
        test_input_match = re.search(r"=== TEST INPUT ===\n(.*?)\nTransform", question, re.DOTALL)

        training_examples = training_examples_match.group(1).strip() if training_examples_match else ""
        test_input = test_input_match.group(1).strip() if test_input_match else ""

        return training_examples, test_input
    except Exception as e:
        return "", ""

def initialize_grid(test_input):
    """Initializes the grid from the test input."""
    try:
        # Remove brackets and split into rows
        rows = test_input.replace("[", "").replace("]", "").split("\n")
        # Split each row into individual numbers and convert to integers
        grid = [list(map(int, row.split(", "))) for row in rows if row]
        return grid
    except Exception as e:
        print(f"Error initializing grid: {e}")
        return None

def adjust_cell_value(grid, row, col, training_examples):
    """Adjusts the cell value based on local context and training examples, with safe string handling."""
    system_instruction = "You are an AI expert in grid transformations."
    prompt = f"""
    You are an AI expert in grid transformations. Given a grid, training examples, the current cell coordinates, and the cell's neighbors, slightly adjust the cell value to better match the patterns seen in the training examples. Only make small adjustments.

    Example 1:
    Training Examples:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[2, 3], [4, 5]]
    Current Grid: [[5, 6], [7, 8]]
    Current Cell: (0, 0) Value: 5
    Neighbors: 6, 7, None, None
    Adjusted Value: 6

    Example 2:
    Training Examples:
    Input Grid: [[1, 0], [0, 1]]
    Output Grid: [[0, 1], [1, 0]]
    Current Grid: [[5, 0], [0, 5]]
    Current Cell: (0, 0) Value: 5
    Neighbors: 0, 0, None, None
    Adjusted Value: 0
    
    Training Examples: {training_examples}
    Current Grid: {grid}
    Current Cell: ({row}, {col}) Value: {grid[row][col]}
    Neighbors: ... # Keep neighbors for now

    Adjusted Value:
    """
    try:
        adjusted_value = call_llm(prompt, system_instruction)
        grid[row][col] = int(adjusted_value.strip())
        return grid
    except Exception as e:
        print(f"Error adjusting cell value: {e}")
        return grid

def enforce_constraints(grid):
    """Enforces explicit constraints on the grid (e.g., row sum)."""
    # Example constraint: Ensure the sum of each row remains constant (if applicable)
    # NOTE: This is a VERY basic constraint. This function must be extended.
    try:
        num_rows = len(grid)
        num_cols = len(grid[0])

        # Check if the grid is valid, we can't process it if it isn't
        if num_rows == 0 or num_cols == 0:
            print("Error: Invalid grid dimensions")
            return grid
        
        # Basic constraint logic
        target_sum = sum(grid[0])  # Try to preserve the sum of the first row as a basic example
        for i in range(num_rows):
            row_sum = sum(grid[i])
            diff = target_sum - row_sum
            # Distribute the difference across the row (very simple)
            if diff != 0:
                grid[i][0] += diff
        return grid
    except Exception as e:
        print(f"Error enforcing constraints: {e}")
        return grid

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response. DO NOT deviate from this example template or invent configuration options. This is how you call the LLM."""
    try:
        from google import genai
        from google.genai import types
        import os

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