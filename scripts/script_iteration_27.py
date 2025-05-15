import os
import re
import math

# EXPLORATION: Meta-Reasoning and Strategy Selection for Grid Transformation with Dynamic Feedback
# HYPOTHESIS: By using a meta-reasoning agent to select an appropriate strategy based on the input question and providing dynamic feedback to refine the chosen strategy, we can improve performance on grid transformation problems. This approach aims to address the core challenge of pattern generalization by dynamically adapting the reasoning process based on the characteristics of each problem.
# The script should identify key structural elements, then create a plan for the next stage based on that strategy

def solve_grid_transformation(question):
    """Solves grid transformation problems by using a meta-reasoning agent and dynamic feedback."""

    # Step 1: Meta-Reasoning and Strategy Selection
    strategy_selection_result = select_strategy(question)
    if not strategy_selection_result["is_valid"]:
        return f"Error: Could not select a strategy. {strategy_selection_result['error']}"

    strategy = strategy_selection_result["strategy"]
    print(f"Chosen strategy: {strategy}") # Add print statement

    # Step 2: Apply Strategy
    transformed_grid = apply_chosen_strategy(question, strategy)
    return transformed_grid

def select_strategy(question):
    """Selects an appropriate strategy for solving the grid transformation problem based on meta-reasoning."""
    system_instruction = "You are a meta-reasoning agent that selects the best strategy for solving grid transformation problems."

    prompt = f"""
    Given the following grid transformation problem, analyze the training examples and select the best strategy for solving it.
    Available Strategies:
    1. Visual Feature Analysis: Analyze visual features such as lines, shapes, patterns, and symmetries to infer the transformation rule.
    2. Pattern Propagation: Identify repeating patterns in the training examples and propagate them to the test input.
    3. Explicit Rule Extraction: Extract explicit rules from the training examples and apply them to the test input.

    Example:
    Problem:
    === TRAINING EXAMPLES ===
    Input Grid:
    [[0, 0, 0],
     [1, 1, 1],
     [0, 0, 0]]
    Output Grid:
    [[2, 2, 2],
     [1, 1, 1],
     [2, 2, 2]]
    Strategy: Pattern Propagation: The "1" pattern remains, while values on the top and bottom get propagated.

    Problem:
    {question}
    Strategy:
    """

    strategy = call_llm(prompt, system_instruction)

    # Validation to ensure *something* was output
    if strategy and strategy.strip():
        return {"is_valid": True, "strategy": strategy, "error": None}
    else:
        return {"is_valid": False, "strategy": None, "error": "Failed to select a strategy."}

def apply_chosen_strategy(question, strategy):
    """Applies the chosen strategy to solve the grid transformation problem."""
    system_instruction = "You are an expert at solving grid transformation problems using a specific strategy."

    if "Visual Feature Analysis" in strategy:
        prompt = f"""
        Given the following grid transformation problem and the chosen strategy (Visual Feature Analysis), analyze the visual features and generate the transformed grid.

        Problem:
        {question}
        Transformed Grid:
        """

    elif "Pattern Propagation" in strategy:
        prompt = f"""
        Given the following grid transformation problem and the chosen strategy (Pattern Propagation), identify repeating patterns and propagate them to the test input.

        Problem:
        {question}
        Transformed Grid:
        """

    elif "Explicit Rule Extraction" in strategy:
        prompt = f"""
        Given the following grid transformation problem and the chosen strategy (Explicit Rule Extraction), extract explicit rules from the training examples and apply them to the test input.

        Problem:
        {question}
        Transformed Grid:
        """

    else:
        return "Error: Invalid strategy chosen."

    transformed_grid = call_llm(prompt, system_instruction)
    return transformed_grid

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

def main(question):
    """Main function to solve the grid transformation task."""
    try:
        answer = solve_grid_transformation(question)
        return answer
    except Exception as e:
        return f"Error in main function: {str(e)}"