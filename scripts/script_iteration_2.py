import os
import re

# Define the call_llm function (provided in the prompt)
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

def extract_grids(text):
    """Extracts all grids from the input text using regex.  Returns a list of grids."""
    grids = re.findall(r'\[\[.*?\]\]', text, re.DOTALL)
    return grids if grids else []


def main(question):
    """
    Main function to solve the grid transformation problem.
    This approach focuses on extracting all grids first, then using the LLM to reason about
    relationships between the grids, and finally applying the transformation.
    NEW HYPOTHESIS: Extracting ALL grids upfront simplifies the reasoning process for the LLM.

    This approach uses:
    1.  Extraction of all grids from the input
    2.  LLM to infer relationships between the training grids.
    3.  LLM to transform the test input based on the inferred relationships.
    4.  LLM to verify the transformation.
    """
    try:
        grids = extract_grids(question)

        if not grids or len(grids) < 3: # Need at least input, output, and test grid
            return "Error: Could not extract enough grids."

        # Infer transformation rule - include all available grids to give the LLM maximum context
        rule_prompt = f"""
        You are an expert at identifying patterns in grid transformations.
        Here are the grids extracted from the problem description:
        {grids}

        Analyze the relationships between these grids to determine the transformation rule.
        Consider all possible relationships.
        Show an example to apply the identified rule to the first Input Grid

        For example:
        Grids: [[[1, 2], [3, 4]], [[2, 4], [6, 8]]]
        Rule: Each cell is multiplied by 2.
        Reasoning: Input Grid values get multiplied by 2 to become the output grid
        Apply Rule: Input [[1, 2], [3, 4]] becomes [[2, 4], [6, 8]]
        """
        transformation_rule = call_llm(rule_prompt)

        # Apply the transformation rule - use explicit examples
        apply_prompt = f"""
        You are an expert at applying grid transformation rules.
        Transformation Rule: {transformation_rule}
        Here is the test input grid that needs to be transformed:
        {grids[-1]}

        Apply this rule to the test input grid. Explain each step.

        For example:
        Rule: Each cell is multiplied by 2.
        Test Input Grid: [[1, 2], [3, 4]]
        Step 1: 1 * 2 = 2
        Step 2: 2 * 2 = 4
        Step 3: 3 * 2 = 6
        Step 4: 4 * 2 = 8
        Transformed Grid: [[2, 4], [6, 8]]

        Apply the transformation to the provided test input grid, and show the reasoning.
        """
        transformed_grid = call_llm(apply_prompt)

        # Verification (NEW: Verify the components in the transformation)
        verification_prompt = f"""
        You are a meticulous grid transformation verifier.
        Transformation Rule: {transformation_rule}
        Test Input Grid: {grids[-1]}
        Transformed Grid: {transformed_grid}

        Verify if the transformed grid follows the transformation rule based on the test input grid.
        Explain your reasoning. Output VALID or INVALID only.

        For example:
        Rule: Each cell is multiplied by 2.
        Test Input Grid: [[1, 2], [3, 4]]
        Transformed Grid: [[2, 4], [6, 8]]
        Reasoning: Each number gets multiplied by 2.
        Result: VALID

        Rule: Each cell is multiplied by 2.
        Test Input Grid: [[1, 2], [3, 4]]
        Transformed Grid: [[2, 4], [6, 9]]
        Reasoning: The bottom right element is incorrect.
        Result: INVALID

        Final Result: Is the grid VALID or INVALID?
        """

        verification_result = call_llm(verification_prompt)

        if "INVALID" in verification_result:
            return f"Error: Verification failed. The grid does not match transformation rule, result: {verification_result}"
        elif "VALID" not in verification_result:
            return f"Error: The grid transformation might be incorrect, result: {verification_result}"
        else:
            # Clean and return
            cleaned_grid = transformed_grid.replace('\n', '').replace(' ', '')
            match = re.search(r'\[.*\]', cleaned_grid)
            if match:
                return match.group(0)
            else:
                return transformed_grid

    except Exception as e:
        return f"Error: {str(e)}"