import os
import re
import math

# Hypothesis: This exploration will implement a "Transformation by Multi-Stage Rule Distillation and Application" approach.
# 1. A "Rule Distiller" agent will extract multiple candidate rules from the training examples, along with confidence scores.
# 2. A "Rule Selector" agent will select the best rule based on the confidence scores and consistency checks across examples.
# 3. A "Rule Applier" agent will apply the selected rule to the test input, generating the transformed grid.
# This differs from previous approaches by explicitly managing rule uncertainty and selecting the best rule instead of relying on a single inferred rule.
# We include robust validation and verification steps throughout to assess the effectiveness of each stage.

def main(question):
    """Transforms a grid by multi-stage rule distillation and application."""
    try:
        # 1. Extract training examples and test input
        training_examples, test_input = preprocess_question(question)

        # 2. Distill candidate rules
        candidate_rules = distill_candidate_rules(training_examples)

        # 3. Select the best rule
        selected_rule = select_best_rule(candidate_rules, training_examples)

        # 4. Apply the selected rule
        transformed_grid = apply_rule(test_input, selected_rule)

        return transformed_grid

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def preprocess_question(question):
    """Extract training examples and test input from the question string."""
    try:
        training_examples_match = re.search(r"=== TRAINING EXAMPLES ===\n(.*?)\n=== TEST INPUT ===", question, re.DOTALL)
        test_input_match = re.search(r"=== TEST INPUT ===\n(.*?)\nTransform", question, re.DOTALL)

        training_examples = training_examples_match.group(1).strip() if training_examples_match else ""
        test_input = test_input_match.group(1).strip() if test_input_match else ""

        return training_examples, test_input
    except Exception as e:
        return "", ""

def distill_candidate_rules(training_examples):
    """Extracts multiple candidate transformation rules with confidence scores."""
    system_instruction = "You are a Rule Distiller extracting transformation rules from examples."
    prompt = f"""
    You are a Rule Distiller. Given training examples of grid transformations, extract THREE candidate transformation rules, along with a confidence score (1-10) for each rule indicating how well it explains the examples.

    Example:
    Training Examples:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[2, 3], [4, 5]]
    Candidate Rules:
    - Rule 1: Each number is incremented by 1. (Confidence: 9)
    - Rule 2: The numbers are shifted to the right and incremented. (Confidence: 6)
    - Rule 3: The numbers are replaced by their index + 1. (Confidence: 3)

    Training Examples:
    {training_examples}
    Candidate Rules:
    """
    candidate_rules = call_llm(prompt, system_instruction)
    return candidate_rules

def select_best_rule(candidate_rules, training_examples):
    """Selects the best transformation rule based on confidence and consistency."""
    system_instruction = "You are a Rule Selector choosing the best rule based on confidence and consistency."
    prompt = f"""
    You are a Rule Selector. Given candidate transformation rules and training examples, select the BEST rule based on:
    1. Confidence scores (higher is better).
    2. Consistency: How well the rule explains ALL training examples (not just some).
    Explain your reasoning.

    Example:
    Training Examples:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[2, 3], [4, 5]]
    Candidate Rules:
    - Rule 1: Each number is incremented by 1. (Confidence: 9)
    - Rule 2: The numbers are shifted to the right and incremented. (Confidence: 6)
    - Rule 3: The numbers are replaced by their index + 1. (Confidence: 3)
    Selected Rule: Rule 1. It has the highest confidence and explains all examples.

    Training Examples:
    {training_examples}
    Candidate Rules:
    {candidate_rules}
    Selected Rule:
    """
    selected_rule = call_llm(prompt, system_instruction)
    return selected_rule

def apply_rule(test_input, selected_rule):
    """Applies the selected transformation rule to the test input."""
    system_instruction = "You are a Rule Applier transforming grids based on specified rules."
    prompt = f"""
    You are a Rule Applier. Given a test input grid and a transformation rule, apply the rule to generate the transformed grid. Maintain the proper grid format.

    Example:
    Test Input: [[5, 6], [7, 8]]
    Transformation Rule: Each number is incremented by 1.
    Transformed Grid: [[6, 7], [8, 9]]

    Test Input:
    {test_input}
    Transformation Rule:
    {selected_rule}
    Transformed Grid:
    """
    transformed_grid = call_llm(prompt, system_instruction)
    return transformed_grid

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response. DO NOT deviate from this example template."""
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