import os
import re

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response"""
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

def analyze_grid_transformation(question, max_attempts=3):
    """Analyzes the grid transformation task using multiple steps with verification."""
    # Hypothesis: Breaking down the task into smaller, verifiable steps will improve accuracy.
    # We'll use separate LLM calls for rule extraction, rule application, and output formatting.

    # Step 1: Extract transformation rule with multiple examples and verification
    rule_extraction_prompt = f"""
    You are an expert in identifying transformation rules from grid examples. Analyze the training examples and explain the transformation rule in a concise sentence.

    Example 1:
    Input Grids:
    Input Grid:
    [[0, 0], [0, 1]]
    Output Grid:
    [[1, 0], [0, 0]]
    Transformation Rule: Swap 0s and 1s.

    Example 2:
    Input Grids:
    [[1, 2], [3, 4]]
    Output Grid:
    [[4, 3], [2, 1]]
    Transformation Rule: Reverse the order of elements in the grid.

    Training Examples from the question:
    {question}

    What is the transformation rule?
    """

    # Implement retry loop with limited attempts
    for attempt in range(max_attempts):
        transformation_rule = call_llm(rule_extraction_prompt, "You are a rule extraction expert.")

        # Verification step: Is the extracted rule coherent?
        verification_prompt = f"""
        You are a critical evaluator. Evaluate the following transformation rule for coherence and completeness based on the training examples.
        If the rule is coherent and complete, respond with 'VALID: [brief explanation]'.
        If the rule has any issues, respond with 'INVALID: [detailed explanation of issues]'.

        Training Examples:
        {question}

        Proposed Transformation Rule:
        {transformation_rule}
        """
        verification_result = call_llm(verification_prompt, "You are a rule verification expert.")

        if verification_result.startswith("VALID:"):
            break # Rule is valid, exit loop
        else:
            print(f"Rule extraction failed (attempt {attempt+1}/{max_attempts}): {verification_result}")

    else:
        transformation_rule = "Could not reliably extract transformation rule." # Default fallback

    # Step 2: Apply the transformation rule to the test input
    test_input = re.search(r'=== TEST INPUT ===\n(.*?)\nTransform', question, re.DOTALL).group(1).strip()
    rule_application_prompt = f"""
    You are an expert at applying transformation rules to grid data. Apply the following transformation rule to the given test input grid.

    Transformation Rule:
    {transformation_rule}

    Test Input Grid:
    {test_input}

    Example Application:
    Rule: Swap 0s and 1s
    Input Grid: [[0, 0], [0, 1]]
    Output Grid: [[1, 1], [1, 0]]

    Apply this rule to generate the output grid. Show ALL reasoning steps.
    """
    transformed_grid = call_llm(rule_application_prompt, "You are an application expert.")

    # Step 3: Format the output grid
    formatting_prompt = f"""
    You are an expert in formatting output grids for a grid transformation task. Convert the following transformed grid into a valid Python list of lists format.

    Transformed Grid:
    {transformed_grid}

    Example Output Format:
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]]

    Ensure that the output is a valid Python list of lists with integer elements, and includes appropriate brackets.
    """
    formatted_grid = call_llm(formatting_prompt, "You are a formatting expert.")

    return formatted_grid

def main(question):
    """Main function to analyze grid transformation tasks."""
    try:
        # Use the analyze_grid_transformation function
        final_answer = analyze_grid_transformation(question)
        return final_answer
    except Exception as e:
        print(f"Error in main: {str(e)}")
        return "I couldn't generate a solution due to an error."