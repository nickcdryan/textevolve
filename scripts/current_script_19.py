import os
import re
import math

# This script solves grid transformation problems using an ensemble of LLM agents, each with a different specialized persona and reasoning style.
# The HYPOTHESIS is that by combining diverse perspectives, we can achieve more robust and accurate solutions.
# It implements the "Ensembling" agentic pattern.

def main(question):
    """Transforms a grid using an ensemble of LLM agents with diverse personas."""
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem by ensembling multiple LLM agents."""

    # Define agent personas
    agent_personas = [
        {"name": "SpatialAnalyst", "instruction": "You are an expert spatial reasoning analyst. You identify geometric patterns and transformations."},
        {"name": "ValueMapper", "instruction": "You are a value mapping specialist. You focus on how values change between input and output grids."},
        {"name": "ConstraintSolver", "instruction": "You are a constraint solver. You ensure that the transformed grid meets all implicit and explicit rules."}
    ]

    # Generate solutions from each agent
    agent_solutions = []
    for persona in agent_personas:
        solution = generate_agent_solution(problem_text, persona)
        agent_solutions.append({"persona": persona["name"], "solution": solution})

    # Aggregate and synthesize the solutions
    final_solution = synthesize_solutions(problem_text, agent_solutions)

    return final_solution

def generate_agent_solution(problem_text, persona):
    """Generates a solution from a single LLM agent with a specific persona."""
    prompt = f"""
    You are a specialized grid transformation agent with the following expertise: {persona["instruction"]}.

    Here's an example of how you should approach the problem:

    Example:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[1, 0], [0, 1]]\n\nOutput Grid:\n[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]\n\n=== TEST INPUT ===\n[[2, 8], [8, 2]]\n\n
    Your Reasoning:
    As a spatial analyst, I see that each element in the input grid becomes a diagonal element in the output grid.

    Transformed Grid:
    [[2, 0, 0, 0], [0, 8, 0, 0], [0, 0, 8, 0], [0, 0, 0, 2]]

    Now, solve the following problem using your specialized expertise:

    Problem: {problem_text}
    """

    return call_llm(prompt, persona["instruction"])

def synthesize_solutions(problem_text, agent_solutions):
    """Synthesizes the solutions from multiple agents into a final solution."""
    solution_strings = "\n".join([f"{s['persona']}: {s['solution']}" for s in agent_solutions])

    prompt = f"""
    You are a solution synthesizer. You are given multiple solutions to the same grid transformation problem, each from a different expert agent.
    Your task is to combine these solutions into a single, comprehensive, and correct final solution.

    Problem: {problem_text}
    Agent Solutions:\n{solution_strings}

    Example:
    Problem: Input [[1,0],[0,1]] , Output [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    SpatialAnalyst: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    ValueMapper: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    ConstraintSolver: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    Synthesized Solution: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

    Synthesized Solution:
    """

    return call_llm(prompt, "You are an expert solution synthesizer.")

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