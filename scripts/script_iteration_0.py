import os
import re

def main(question):
    """
    Solve grid transformation problems using a multi-stage LLM approach with explicit rule extraction and verification.
    This approach aims to explicitly extract transformation rules and then apply them.
    """

    # HYPOTHESIS: Explicitly extracting and verifying transformation rules will improve accuracy and robustness.
    # We will test this by adding verification steps after rule extraction and application.

    try:
        # 1. Extract transformation rules with examples
        rules = extract_transformation_rules(question)

        # 2. Verify the extracted rules
        rule_verification = verify_transformation_rules(question, rules)
        if not rule_verification["is_valid"]:
            return "Error: Extracted rules are invalid. " + rule_verification["feedback"]

        # 3. Apply the rules to transform the test input
        transformed_grid = apply_transformation_rules(question, rules)

        # 4. Verify the transformed grid
        grid_verification = verify_transformed_grid(question, transformed_grid)
        if not grid_verification["is_valid"]:
            return "Error: Transformed grid is invalid. " + grid_verification["feedback"]

        return transformed_grid

    except Exception as e:
        return f"Error: {str(e)}"

def extract_transformation_rules(question):
    """Extract transformation rules from the question using the LLM."""
    system_instruction = "You are an expert in identifying patterns and extracting transformation rules from grid examples."

    prompt = f"""
    Analyze the grid transformation examples and extract the underlying rules.

    Example 1:
    Input Grid:
    [[0, 1, 0], [1, 1, 0], [0, 1, 0]]
    Output Grid:
    [[0, 2, 0], [2, 2, 0], [0, 2, 0]]
    Rules: Replace 1 with 2.

    Example 2:
    Input Grid:
    [[1, 2], [3, 4]]
    Output Grid:
    [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]
    Rules: Replicate rows and columns.

    Now, analyze the transformation in the following question:
    {question}

    Provide the extracted rules in a concise format.
    """
    return call_llm(prompt, system_instruction)

def verify_transformation_rules(question, rules):
    """Verify if the extracted rules are consistent with the training examples."""
    system_instruction = "You are a validator who checks if the extracted rules are consistent with the given examples."

    prompt = f"""
    Verify if the following rules accurately describe the transformations in the given examples:

    Question: {question}
    Extracted Rules: {rules}

    Example validation 1:
    Question: Grid Transformation Task
    === TRAINING EXAMPLES ===
    Input Grid: [[0, 1, 0], [1, 1, 0], [0, 1, 0]]
    Output Grid: [[0, 2, 0], [2, 2, 0], [0, 2, 0]]
    Extracted Rules: Replace 1 with 2.
    Validation: Valid

    Example validation 2:
    Question: Grid Transformation Task
    === TRAINING EXAMPLES ===
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]
    Extracted Rules: Replicate rows and columns.
    Validation: Valid

    Determine if the extracted rules accurately describe the transformations in the provided question.
    Return a JSON-like structure: {{"is_valid": true/false, "feedback": "explanation"}}
    """

    verification_result = call_llm(prompt, system_instruction)
    #Attempt to extract relevant fields from LLM response text
    is_valid = "Valid" in verification_result
    feedback = verification_result
    
    return {"is_valid": is_valid, "feedback": feedback}

def apply_transformation_rules(question, rules):
    """Apply the extracted rules to transform the test input grid."""
    system_instruction = "You are an expert at applying transformation rules to grids."

    prompt = f"""
    Apply the following transformation rules to the test input grid in the question:

    Question: {question}
    Transformation Rules: {rules}

    Example application:
    Question: Grid Transformation Task
    === TRAINING EXAMPLES ===
    Input Grid: [[0, 1, 0], [1, 1, 0], [0, 1, 0]]
    Output Grid: [[0, 2, 0], [2, 2, 0], [0, 2, 0]]
    === TEST INPUT ===
    [[1, 1, 1], [0, 1, 0], [0, 1, 0]]
    Transformation Rules: Replace 1 with 2.
    Transformed Grid: [[2, 2, 2], [0, 2, 0], [0, 2, 0]]

    Apply the transformation to the test input.
    Provide the transformed grid as a nested list.
    """
    return call_llm(prompt, system_instruction)

def verify_transformed_grid(question, transformed_grid):
    """Verify if the transformed grid is consistent with the extracted rules and examples."""
    system_instruction = "You are a validator who checks if the transformed grid follows the transformation rules."

    prompt = f"""
    Verify if the transformed grid is consistent with the transformation rules and examples.

    Question: {question}
    Transformed Grid: {transformed_grid}

    Example validation:
    Question: Grid Transformation Task
    === TRAINING EXAMPLES ===
    Input Grid: [[0, 1, 0], [1, 1, 0], [0, 1, 0]]
    Output Grid: [[0, 2, 0], [2, 2, 0], [0, 2, 0]]
    === TEST INPUT ===
    [[1, 1, 1], [0, 1, 0], [0, 1, 0]]
    Transformed Grid: [[2, 2, 2], [0, 2, 0], [0, 2, 0]]
    Validation: Valid

    Determine if the transformed grid is valid.
    Return a JSON-like structure: {{"is_valid": true/false, "feedback": "explanation"}}
    """

    verification_result = call_llm(prompt, system_instruction)
    #Attempt to extract relevant fields from LLM response text
    is_valid = "Valid" in verification_result
    feedback = verification_result
    
    return {"is_valid": is_valid, "feedback": feedback}