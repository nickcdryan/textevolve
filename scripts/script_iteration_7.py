import os
import re
import math
from google import genai
from google.genai import types

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
    """
    EXPLORATION: This script implements a **Multi-Agent Debate with Verification Loop**.

    HYPOTHESIS: By simulating a debate between two specialized agents (a "Solution Proposer" and a "Solution Critic") and including a verification loop,
    we can achieve more robust and accurate answers compared to previous single-agent or simpler multi-stage approaches.
    The debate encourages deeper analysis and challenges assumptions, while the verification loop ensures adherence to the problem's constraints.

    This is a RADICAL departure from all past approaches, focusing on internal model debate rather than external knowledge retrieval, decomposition, or strict extraction.
    """

    # Agent 1: Solution Proposer
    def propose_solution(problem, attempt):
        prompt = f"""
        You are a Solution Proposer. Your goal is to provide a concise and accurate answer to the given problem.
        This is attempt {attempt} to solve the problem.

        Example 1:
        Problem: Which player kicked the only field goal of the game?
        Proposed Solution: Josh Scobee

        Example 2:
        Problem: Which star has a smaller mass, Nu Phoenicis or Gliese 915?
        Proposed Solution: Gliese 915

        Problem: {problem}
        Proposed Solution:
        """
        return call_llm(prompt, "You are an expert problem solver proposing solutions.")

    # Agent 2: Solution Critic
    def critique_solution(problem, solution, attempt):
        prompt = f"""
        You are a Solution Critic. Your goal is to rigorously evaluate the proposed solution and identify any potential flaws, inaccuracies, or inconsistencies.
        This is attempt {attempt} to critique the solution.

        Example 1:
        Problem: Which player kicked the only field goal of the game?
        Proposed Solution: Tom Brady
        Critique: Tom Brady is a quarterback, not a kicker. This solution is incorrect.

        Example 2:
        Problem: Which star has a smaller mass, Nu Phoenicis or Gliese 915?
        Proposed Solution: Gliese 915
        Critique: This solution appears correct. Gliese 915 is indeed smaller.

        Problem: {problem}
        Proposed Solution: {solution}
        Critique:
        """
        return call_llm(prompt, "You are an expert at critiquing solutions and finding flaws.")

    # Verification Loop with Debate
    solution = ""
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        solution = propose_solution(question, attempt)
        critique = critique_solution(question, solution, attempt)

        # Verification: Check if the critique identifies any major issues
        if "incorrect" in critique.lower() or "flaw" in critique.lower():
            # Debate continues: Proposer adjusts the solution based on critique
            solution = call_llm(f"""
            Based on the critic's comments: '{critique}', refine your solution to the problem: '{question}'.
            Previous solution: '{solution}'
            """, "You are refining your solution based on expert critique.")
        else:
            # No major issues found: Verification passes, solution is deemed acceptable
            break # Exit loop, solution is deemed acceptable

    # Final Answer: The solution after the debate and verification loop
    final_answer = solution

    # Diagnostic Print Statements
    print(f"Initial Question: {question}")
    print(f"Final Answer: {final_answer}")

    return final_answer