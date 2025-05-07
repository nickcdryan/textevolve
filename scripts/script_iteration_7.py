import os
import re

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

def analyze_grid_transformation(question, max_attempts=3):
    """Analyzes grid transformation problems by generating multiple possible rules and selecting the best one."""

    # Hypothesis: Generating multiple candidate rules and scoring them based on their ability to explain the example grids improves accuracy.

    # Step 1: Generate multiple candidate rules
    rule_generation_prompt = f"""
    Analyze the following grid transformation problem and generate three possible transformation rules. Consider different types of transformations (e.g., rotations, reflections, color swaps).

    Example:
    Input Grids:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[4, 3], [2, 1]]
    Possible Rules:
    1. Reverse the order of all elements.
    2. Swap top-left with bottom-right and top-right with bottom-left.
    3. Rotate 180 degrees.

    Problem:
    {question}

    Possible Rules:
    1. 
    2. 
    3. 
    """
    candidate_rules_text = call_llm(rule_generation_prompt, system_instruction="You are an expert at generating possible rules for grid transformations. Provide three distinct rules.")
    candidate_rules = [rule.strip() for rule in candidate_rules_text.split('\n') if rule.strip()]

    # Step 2: Score each rule based on its ability to explain the training examples
    scoring_prompt = f"""
    Evaluate each transformation rule based on its ability to explain the training examples. Assign a score (1-10) based on how well the rule accounts for the transformation from input to output.

    Problem:
    {question}
    Candidate Rules:
    {candidate_rules_text}

    Evaluation:
    """
    rule_scores_text = call_llm(scoring_prompt, system_instruction="You are a rule evaluation expert. Assign scores (1-10) to each rule based on its explanatory power.")
    rule_scores = [int(re.search(r'(\d+)', score).group(1)) for score in rule_scores_text.split('\n') if re.search(r'(\d+)', score)]

    # Step 3: Select the best rule based on the scores
    best_rule_index = rule_scores.index(max(rule_scores))
    best_rule = candidate_rules[best_rule_index]

    # Step 4: Apply the best rule to the test input
    test_input = re.search(r'=== TEST INPUT ===\n(.*?)\nTransform', question, re.DOTALL).group(1).strip()
    application_prompt = f"""
    Apply the following transformation rule to the given test input grid.

    Rule:
    {best_rule}
    Test Input Grid:
    {test_input}

    Output Grid:
    """
    transformed_grid = call_llm(application_prompt, "You are an application expert.")

    # Step 5: Format the output grid
    formatting_prompt = f"""
    Convert the transformed grid into a valid Python list of lists format.

    Transformed Grid:
    {transformed_grid}

    Example Output Format:
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    """
    formatted_grid = call_llm(formatting_prompt, "You are a formatting expert.")

    return formatted_grid

def main(question):
    """Main function to process the grid transformation question."""
    try:
        answer = analyze_grid_transformation(question)
        return answer
    except Exception as e:
        return f"Error: {str(e)}"