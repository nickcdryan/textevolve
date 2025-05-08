import os
import re
import math

# This script takes a radically different approach: rather than focusing on explicit rule extraction,
# it uses the LLM to perform iterative "hallucination" and refinement of the grid, based on overall patterns.
# The hypothesis is that the LLM can, through iterative feedback, converge on a valid solution WITHOUT needing to
# explicitly articulate the transformation rule. We will use the concept of LLM Self-Consistency (CoT-SC) to iteratively
# refine the output by providing the previous outputs back as context.

def main(question):
    """Transforms a grid using LLM-driven iterative refinement, without explicit rule extraction."""
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=5):
    """Solves the grid transformation problem through iterative refinement and feedback."""
    system_instruction = "You are an expert at grid transformation, able to refine and adjust your solution iteratively."

    # STEP 1: Initial Grid Hallucination
    initial_hallucination_prompt = f"""
    You are presented with a grid transformation problem. Generate a possible transformed grid, based on your understanding of common grid patterns.
    Assume a 3x3 grid output if you are unsure, but attempt to match output grid to input grid size.

    Problem: {problem_text}

    Example 1:
    Problem: Input Grid: [[1, 0], [0, 1]] Output Grid: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    Hallucination: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    Example 2:
    Problem: Input Grid: [[2, 8], [8, 2]] Output Grid: [[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]
    Hallucination: [[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]

    Hallucination:
    """

    current_grid = call_llm(initial_hallucination_prompt, system_instruction)
    print(f"Initial hallucination: {current_grid}")

    # STEP 2: Iterative Refinement and Feedback
    for attempt in range(max_attempts):
        refinement_prompt = f"""
        You are tasked with refining the transformation for this grid-based problem.
        Here is your previous attempt, along with the original problem. Carefully adjust your approach. Focus on spatial relationships and matching demonstrated patterns in the training examples.
        Original Problem: {problem_text}
        Previous Attempt: {current_grid}

        Example 1:
        Original Problem: Input Grid: [[1, 0], [0, 1]] Output Grid: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        Previous Attempt: [[1, 0], [0, 1]]
        Refinement: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

        Example 2:
        Original Problem: Input Grid: [[2, 8], [8, 2]] Output Grid: [[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]
        Previous Attempt: [[2, 8], [8, 2]]
        Refinement: [[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]

        Refinement:
        """

        current_grid = call_llm(refinement_prompt, system_instruction)
        print(f"Refinement attempt {attempt + 1}: {current_grid}")

        # Step 3: Answer Checker - validates if the answer is a valid transformed grid that fits the problem
        answer_check_prompt = f"""
        You need to validate if the given answer follows the rules of the grid-based transformation problem and can be taken as a valid answer.

        Problem: {problem_text}
        Proposed Solution: {current_grid}

        If the answer is a transformed grid as the solution to the problem, respond with VALID
        If the answer is not a valid grid solution to the problem, respond with INVALID

        Validity:
        """

        validity = call_llm(answer_check_prompt, system_instruction)
        if "VALID" in validity:
            return current_grid

    # If we've failed after max_attempts, return a default grid as a last resort.
    return "[[0,0,0],[0,0,0],[0,0,0]]"

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