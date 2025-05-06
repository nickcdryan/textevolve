import os
import re

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response.  DO NOT deviate from this example template or invent configuration options. This is how you call the LLM."""
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

def extract_grid(text):
    """Extract the grid from the input text using regex."""
    match = re.search(r'\[.*?\n.*?\]', text, re.DOTALL)
    if match:
        return match.group(0)
    return None

def main(question):
    """
    Main function to solve the grid transformation problem.
    This approach uses a multi-stage LLM pipeline to:
    1. Extract the input and training grids.
    2. Infer the transformation rule.
    3. Apply the rule to the test input.
    4. Verify the result.
    """
    try:
        # 1. Extract grids
        input_grid_text = extract_grid(question)
        if not input_grid_text:
            return "Error: Could not extract input grid."

        # Extract training examples - attempt to get two, if availabe
        example1_start = question.find("Example 1:")
        example2_start = question.find("Example 2:")
        example_end = question.find("=== TEST INPUT ===")

        if example1_start != -1 and example2_start != -1 and example_end != -1:
           training_examples = question[example1_start:example_end]
        elif example1_start != -1 and example_end != -1:
           training_examples = question[example1_start:example_end]
        else:
           training_examples = "No training examples found"
           return "Error: No training examples found"


        # 2. Infer transformation rule
        rule_prompt = f"""
        You are an expert at identifying patterns in grid transformations.

        Here are examples of grid transformations:
        {training_examples}

        Based on these examples, describe the transformation rule.
        Consider patterns like:
        - Expansion/contraction of the grid
        - Value changes based on position
        - Relationships between neighboring cells

        Example:
        Input:
        Input Grid:
        [[1, 2], [3, 4]]
        Output Grid:
        [[2, 4], [6, 8]]
        Reasoning: Each cell is multiplied by 2.
        Output: Each cell is multiplied by 2.

        What is the transformation rule?
        """
        transformation_rule = call_llm(rule_prompt)

        # 3. Apply the rule
        apply_prompt = f"""
        You are an expert at applying grid transformation rules.
        Transformation Rule: {transformation_rule}
        Apply this rule to the following input grid:
        {input_grid_text}

        Example:
        Transformation Rule: Each cell is multiplied by 2.
        Input Grid:
        [[1, 2], [3, 4]]
        Output Grid:
        [[2, 4], [6, 8]]

        Apply the rule and output the resulting grid.
        """
        transformed_grid = call_llm(apply_prompt)

        # 4. Verification (NEW HYPOTHESIS: Use LLM as a verifier)
        verification_prompt = f"""
        You are a meticulous grid transformation verifier.
        Question: {question}
        Transformation Rule: {transformation_rule}
        Transformed Grid: {transformed_grid}
        Verify if the transformed grid follows the transformation rule based on training examples.
        If the transformation looks good, say "VALID". Otherwise, if the resulting transformation is incorrect or doesn't follow the rule, say "INVALID".
        Example:
        Question:
        Grid Transformation Task

        === TRAINING EXAMPLES ===

        Example 1:
        Input Grid:
        [[1, 0], [0, 1]]

        Output Grid:
        [[2, 0], [0, 2]]
        Transformation Rule: Every 1 becomes 2

        Transformed Grid: [[2, 0], [0, 2]]
        Result: VALID

        Question:
        Grid Transformation Task

        === TRAINING EXAMPLES ===

        Example 1:
        Input Grid:
        [[1, 0], [0, 1]]

        Output Grid:
        [[2, 0], [0, 2]]
        Transformation Rule: Every 1 becomes 2

        Transformed Grid: [[2, 0], [0, 1]]
        Result: INVALID

        Final Result: Is the grid valid or invalid?
        """

        verification_result = call_llm(verification_prompt)
        if "INVALID" in verification_result:
            return f"Error: Verification failed. The grid does not match transformation rule, result: {verification_result}"
        elif "VALID" not in verification_result:
            return f"The grid transformation might be incorrect, result: {verification_result}"
        else:

            # 5. Clean the output
            cleaned_grid = transformed_grid.replace('\n', '').replace(' ', '')
            match = re.search(r'\[.*\]', cleaned_grid)
            if match:
                return match.group(0)
            else:
               return transformed_grid


    except Exception as e:
        return f"Error: {str(e)}"