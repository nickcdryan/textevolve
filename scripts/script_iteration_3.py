import os
import re
import math

def main(question):
    """
    Main function to solve the grid transformation task.
    This approach uses a decomposition into specialized agents: Data Extraction, Pattern Analyzer, and Transformation Applier,
    with validation loops after each major step.

    Hypothesis: By using specialized agents and explicit validation, we can improve robustness and accuracy.
    """
    try:
        return solve_grid_transformation(question)
    except Exception as e:
        return f"Error: {str(e)}"

def solve_grid_transformation(question):
    """
    Solves the grid transformation problem using specialized agents and validation.
    """

    # --- Agent 1: Data Extraction Agent ---
    def extract_data(problem_description, max_attempts=3):
        """Extracts training examples and test input from the problem description."""
        system_instruction = "You are a precise data extraction specialist. Your goal is to extract structured data accurately."
        prompt = f"""
        Extract the training examples and test input from the following problem description.

        Example 1:
        Problem Description:
        Training Examples:
        [{{\"input\":[[1,2],[3,4]],\"output\":[[5,6],[7,8]]}}]
        Test Input:
        [[9,10],[11,12]]
        Extracted Data:
        {{
          "training_examples": "[{{\\"input\\":[[1,2],[3,4]],\\"output\\":[[5,6],[7,8]]}}]",
          "test_input": "[[9,10],[11,12]]"
        }}

        Example 2:
        Problem Description:
        Training Examples:
        [{{\"input\":[[0,1],[1,0]],\"output\":[[2,3],[3,2]]}}, {{\"input\":[[1,1],[0,0]],\"output\":[[3,3],[2,2]]}}]
        Test Input:
        [[0,0],[1,1]]
        Extracted Data:
        {{
          "training_examples": "[{{\\"input\\":[[0,1],[1,0]],\\"output\\":[[2,3],[3,2]]}}, {{\"input\":[[1,1],[0,0]],\\"output\\":[[3,3],[2,2]]}}]",
          "test_input": "[[0,0],[1,1]]"
        }}

        Problem Description:
        {problem_description}
        Extracted Data:
        """

        for attempt in range(max_attempts):
            extracted_data_str = call_llm(prompt, system_instruction)
            try:
                training_examples = re.search(r'"training_examples":\s*"([^"]*)"', extracted_data_str).group(1)
                test_input = re.search(r'"test_input":\s*"([^"]*)"', extracted_data_str).group(1)
                return {"training_examples": training_examples, "test_input": test_input, "is_valid": True}
            except Exception as e:
                if attempt < max_attempts - 1:
                    print(f"Extraction failed, retrying: {e}")
                    continue
                return {"error": str(e), "is_valid": False}
        return {"error": "Max extraction attempts exceeded", "is_valid": False}

    extraction_result = extract_data(question)

    if not extraction_result["is_valid"]:
        return f"Data extraction error: {extraction_result.get('error', 'Unknown error')}"

    training_examples = extraction_result["training_examples"]
    test_input = extraction_result["test_input"]

    # --- Agent 2: Pattern Analyzer Agent ---
    def analyze_pattern(training_data, max_attempts=3):
        """Analyzes training examples to identify the transformation pattern."""
        system_instruction = "You are an expert pattern analyzer for grid transformations. Identify patterns, not code."
        prompt = f"""
        Analyze the training examples to identify the transformation pattern. Describe the pattern in words, NOT in code.

        Example 1:
        Training Examples:
        [{{\"input\":[[1,2],[3,4]],\"output\":[[5,6],[7,8]]}}]
        Transformation Pattern:
        Each element in the input grid is increased by 4 to obtain the corresponding element in the output grid.

        Example 2:
        Training Examples:
        [{{\"input\":[[0,1],[1,0]],\"output\":[[2,3],[3,2]]}}, {{\"input\":[[1,1],[0,0]],\"output\":[[3,3],[2,2]]}}]
        Transformation Pattern:
        Each element in the input grid is increased by 2 to obtain the corresponding element in the output grid.

        Training Examples:
        {training_data}
        Transformation Pattern:
        """
        for attempt in range(max_attempts):
            transformation_pattern = call_llm(prompt, system_instruction)
            if transformation_pattern:
                return {"pattern": transformation_pattern, "is_valid": True}
            if attempt < max_attempts - 1:
                continue
            return {"error": "Max pattern analysis attempts exceeded", "is_valid": False}

    pattern_result = analyze_pattern(training_examples)
    if not pattern_result["is_valid"]:
        return f"Pattern analysis error: {pattern_result.get('error', 'Unknown error')}"

    transformation_pattern = pattern_result["pattern"]

    # --- Agent 3: Transformation Applier Agent ---
    def apply_transformation(pattern, input_grid, max_attempts=3):
        """Applies the transformation pattern to the test input."""
        system_instruction = "You are a transformation expert. Apply transformation patterns to input grids."
        prompt = f"""
        Apply the following transformation pattern to the test input. Provide the transformed output grid.

        Transformation Pattern:
        {pattern}
        Test Input:
        {input_grid}

        Example:
        Transformation Pattern:
        Each element in the input grid is increased by 4 to obtain the corresponding element in the output grid.
        Test Input:
        [[9,10],[11,12]]
        Transformed Output:
        [[13,14],[15,16]]

        Transformed Output:
        """
        for attempt in range(max_attempts):
            transformed_output = call_llm(prompt, system_instruction)

            # Validation using Regex: Valid output should be a 2D array string
            pattern = r'^(\[\[\d+(,\s*\d+)*\](,\s*\[\d+(,\s*\d+)*\])*\])$'
            if re.match(pattern, transformed_output):
                return {"transformed_output": transformed_output, "is_valid": True}
            else:
                if attempt < max_attempts - 1:
                    print(f"Apply transformation failed, retrying {attempt}: Invalid format")
                    continue
                return {"error": f"Max transform attempts exceeded: Invalid format", "is_valid": False}
        return {"error": "Max transform attempts exceeded", "is_valid": False}

    transform_result = apply_transformation(transformation_pattern, test_input)
    if not transform_result["is_valid"]:
        return f"Transformation application error: {transform_result.get('error', 'Unknown error')}"

    final_output = transform_result["transformed_output"]

    return final_output
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