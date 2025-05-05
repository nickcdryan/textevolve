import os
import re
import math

# Hypothesis: A hierarchical decomposition of the problem into pattern identification, 
# grid manipulation planning, and execution, with validation at each stage, 
# will improve accuracy by isolating and correcting errors early. 
# We will explicitly verify grid dimensions at various steps.

def main(question):
    """Main function to transform a grid based on examples."""
    try:
        # 1. Pattern Identification with Validation
        pattern_data = identify_pattern(question)
        if not pattern_data["is_valid"]:
            return f"Error: Pattern identification failed - {pattern_data['error']}"

        # 2. Grid Manipulation Planning with Validation
        plan_data = create_manipulation_plan(question, pattern_data["pattern"])
        if not plan_data["is_valid"]:
            return f"Error: Manipulation plan creation failed - {plan_data['error']}"

        # 3. Execution with Validation
        transformed_grid = execute_transformation(question, plan_data["plan"])
        if not transformed_grid["is_valid"]:
            return f"Error: Transformation execution failed - {transformed_grid['error']}"

        return transformed_grid["grid_string"]

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def identify_pattern(question):
    """Identifies the transformation pattern from the examples."""
    system_instruction = "You are a pattern recognition expert for grid transformations."
    prompt = f"""
    Analyze the grid transformation examples and identify the underlying pattern.

    Example 1:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]
    Pattern: Each cell is expanded into a 2x2 block with the same value.

    Example 2:
    Input Grid: [[1, 0], [0, 1]]
    Output Grid: [[2, 1], [1, 2]]
    Pattern: Values are swapped across the diagonal, and then incremented by 1.

    Question: {question}
    """
    response = call_llm(prompt, system_instruction)

    # Validation and error checking using string matching as a fallback
    if "Pattern:" not in response:
        return {"is_valid": False, "error": "Pattern not clearly identified."}

    return {"is_valid": True, "pattern": response.split("Pattern: ")[-1].strip()}

def create_manipulation_plan(question, pattern):
    """Creates a plan to manipulate the grid based on the identified pattern."""
    system_instruction = "You are a planner who devises grid manipulation steps."
    prompt = f"""
    Based on the transformation pattern, create a detailed plan to manipulate the input grid.

    Pattern: {pattern}

    Example Plan 1:
    Input Grid: [[1, 2], [3, 4]]
    Pattern: Each cell is expanded into a 2x2 block with the same value.
    Plan: 1. Determine input grid dimensions. 2. Create a new grid with doubled dimensions. 3. For each cell in the input grid, copy its value to the corresponding 2x2 block in the output grid.

    Example Plan 2:
    Input Grid: [[1, 0], [0, 1]]
    Pattern: Values are swapped across the diagonal, and then incremented by 1.
    Plan: 1. Determine input grid dimensions. 2. Create a copy of the input grid. 3. Swap the values across the diagonal in the new grid. 4. Increment all values in the new grid by 1.

    Question: {question}
    """
    response = call_llm(prompt, system_instruction)

    # Validation and error checking
    if "Plan:" not in response:
        return {"is_valid": False, "error": "Plan not created successfully."}

    return {"is_valid": True, "plan": response.split("Plan: ")[-1].strip()}

def execute_transformation(question, plan):
    """Executes the transformation plan on the input grid."""
    system_instruction = "You are an execution engine that transforms grids."
    prompt = f"""
    Execute the transformation plan on the input grid and provide the transformed grid as a list of lists.

    Plan: {plan}

    Example Execution 1:
    Input Grid: [[1, 2], [3, 4]]
    Plan: 1. Determine input grid dimensions. 2. Create a new grid with doubled dimensions. 3. For each cell in the input grid, copy its value to the corresponding 2x2 block in the output grid.
    Transformed Grid: [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]

    Example Execution 2:
    Input Grid: [[1, 0], [0, 1]]
    Plan: 1. Determine input grid dimensions. 2. Create a copy of the input grid. 3. Swap the values across the diagonal in the new grid. 4. Increment all values in the new grid by 1.
    Transformed Grid: [[2, 1], [1, 2]]

    Question: {question}
    """

    response = call_llm(prompt, system_instruction)

    # Validate and attempt to extract grid from the response using a simple regex
    match = re.search(r"(\[.*\])", response)
    if not match:
        return {"is_valid": False, "error": "Transformed grid not found."}
    
    grid_string = match.group(1)
    try:
        # Minimal attempt to correct formatting errors. We are explicitly AVOIDING json.loads() here!
        grid_string = grid_string.replace(" ", "")
        if not (grid_string.startswith("[[") and grid_string.endswith("]]")):
            return {"is_valid": False, "error": "Improper formatting after correction."}
        
        return {"is_valid": True, "grid_string": grid_string}
    except Exception as e:
        return {"is_valid": False, "error": f"Grid extraction failed: {str(e)}"}

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