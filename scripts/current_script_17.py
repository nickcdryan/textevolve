import os
import re
import math

# This script takes a radically different approach. It focuses on using the LLM to perform iterative "self-correction" of the *entire* reasoning process, rather than individual steps.
# The hypothesis is that by prompting the LLM to review its entire chain of thought from extraction to transformation and identifying inconsistencies, we can drive the LLM to fix its own errors.
# We will prompt the LLM to review the entire history (problem, reasoning, output), then use this information to generate a new, corrected solution.

def main(question):
    """Transforms a grid using LLM-driven iterative refinement of entire reasoning process."""
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem through iterative refinement of the entire reasoning chain and feedback."""
    system_instruction = "You are an expert at grid transformation, able to identify and correct errors in your reasoning and solution iteratively."

    # STEP 1: Initial Solution Generation - as before
    initial_solution_prompt = f"""
    You are presented with a grid transformation problem.

    Problem: {problem_text}

    Example 1:
    Problem: Input Grid: [[1, 0], [0, 1]] Output Grid: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    Solution: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    Example 2:
    Problem: Input Grid: [[2, 8], [8, 2]] Output Grid: [[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]
    Solution: [[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]

    Provide a solution to this new problem.
    """

    current_solution = call_llm(initial_solution_prompt, system_instruction)
    full_reasoning_chain = f"Problem: {problem_text}\nInitial Solution: {current_solution}"

    # STEP 2: Iterative Self-Correction with Reasoning Chain Review
    for attempt in range(max_attempts):
        review_prompt = f"""
        You are an expert at reviewing your own work. You are given the FULL history of your attempt to solve a grid transformation problem.
        Your task is to identify any errors in your reasoning or solution, and then generate a completely new, corrected solution.

        Full Reasoning Chain:
        {full_reasoning_chain}

        Example 1:
        Full Reasoning Chain: Problem: Input [[1, 0], [0, 1]] Initial Solution: [[1, 0], [0, 0]]
        Critique: The solution is WRONG. It only copied the first row. It should have created a diagonal pattern.
        Corrected Solution: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

        Example 2:
        Full Reasoning Chain: Problem: Input [[2, 8], [8, 2]] Initial Solution: [[2, 8], [8, 2]]
        Critique: The solution is WRONG. It only copied the input. Each element should have expanded to a 2x2 block.
        Corrected Solution: [[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]

        Critique your reasoning and provide a new, fully corrected solution:
        """

        corrected_response = call_llm(review_prompt, system_instruction)
        # Split the response to keep full reasoning chain.
        try:
            critique = corrected_response.split("Corrected Solution:")[0]
            current_solution = corrected_response.split("Corrected Solution:")[1]
        except:
            critique = corrected_response # If the model does not add "Corrected Solution:", it has just critiqued the response.
            current_solution = current_solution # The model has only critiqued the response. Keep the same current solution.
        
        full_reasoning_chain += f"\nCritique (Attempt {attempt+1}): {critique}\nCorrected Solution (Attempt {attempt+1}): {current_solution}"
        
        #Basic check to prevent early break.
        if "VALID" in current_solution:
          return current_solution

    # If we've failed after max_attempts, return a default grid as a last resort.
    return "[[0,0,0],[0,0,0],[0,0,0]]"

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response."""
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