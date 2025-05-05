import os
import re
import math

# Hypothesis: This exploration will implement a "Transformation by Visual-Spatial Attention and Rule-Guided Diffusion" approach.
# This approach first converts the grid into a symbolic visual representation with explicit spatial coordinates, uses multi-example prompting for rule distillation based on visual patterns, and uses an iterative diffusion process guided by these rules to gradually transform the test grid.
# It aims to enhance spatial understanding and rule generalization.

def main(question):
    """Transforms a grid by visual-spatial attention and rule-guided diffusion."""
    try:
        # 1. Extract training examples and test input
        training_examples, test_input = preprocess_question(question)

        # 2. Convert test input grid to symbolic visual representation
        symbolic_grid = grid_to_symbolic(test_input)

        # 3. Distill transformation rule with visual spatial attention
        transformation_rule = distill_transformation_rule(training_examples, symbolic_grid)

        # 4. Apply transformation using rule-guided diffusion
        transformed_grid = rule_guided_diffusion(symbolic_grid, transformation_rule)

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

def grid_to_symbolic(grid_str):
    """Convert the grid into a symbolic visual representation with coordinates."""
    try:
        grid = eval(grid_str)
        symbolic_repr = ""
        for r, row in enumerate(grid):
            for c, val in enumerate(row):
                symbolic_repr += f"({r},{c}):{val} "
            symbolic_repr += "\n"
        return symbolic_repr
    except Exception as e:
        return f"Error converting to symbolic representation: {str(e)}"

def distill_transformation_rule(training_examples, symbolic_grid):
    """Distill the transformation rule with visual spatial attention."""
    system_instruction = "You are an expert at identifying visual-spatial patterns and formulating transformation rules."
    prompt = f"""
    You are an expert at identifying visual-spatial patterns in grid transformations and formulating transformation rules.

    Example 1:
    Training Examples:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[2, 3], [4, 5]]
    Symbolic Grid: (0,0):1 (0,1):2 
                   (1,0):3 (1,1):4 
    Transformation Rule: Each value is incremented by 1.

    Example 2:
    Training Examples:
    Input Grid: [[0, 1], [1, 0]]
    Output Grid: [[1, 0], [0, 1]]
    Symbolic Grid: (0,0):0 (0,1):1 
                   (1,0):1 (1,1):0 
    Transformation Rule: Values are swapped diagonally.

    Training Examples:
    {training_examples}
    Symbolic Grid:
    {symbolic_grid}
    Transformation Rule:
    """
    transformation_rule = call_llm(prompt, system_instruction)
    return transformation_rule

def rule_guided_diffusion(symbolic_grid, transformation_rule):
    """Apply the transformation using rule-guided diffusion."""
    system_instruction = "You are an AI that applies rules gradually to a symbolic grid to converge on a final transformed output."
    prompt = f"""
    You are an AI that applies transformation rules gradually to a symbolic grid. Your goal is to converge on a final transformed output following the given rule.

    Symbolic Grid:
    {symbolic_grid}

    Transformation Rule:
    {transformation_rule}

    Apply the rule step-by-step, showing each intermediate state, until the grid is fully transformed. Output the final transformed grid in the format: [[...],[...]]
    """
    transformed_grid = call_llm(prompt, system_instruction)
    return transformed_grid

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