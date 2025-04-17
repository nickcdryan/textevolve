import os
import json
import re
import math

def main(question):
    """
    Schedules meetings by dynamically routing input to specialized handlers, then verifies the solution using a dedicated agent.

    This approach uses dynamic routing and a verification agent, differing from previous sequential or ReAct approaches.
    """
    try:
        analysis = analyze_input(question) # Analyze input for routing
        if "Error" in analysis: return analysis

        if "schedule" in analysis.lower():
            solution = schedule_meeting(question) # Route to scheduler
            if "Error" in solution: return solution
        else:
            return "Input not recognized as scheduling request."

        verification = verify_solution(question, solution) # Verify solution
        return verification
    except Exception as e:
        return f"Error in main: {str(e)}"

def analyze_input(question):
    """Analyzes the input to determine its type using LLM with embedded example."""
    system_instruction = "You are an input analyzer. Determine if the input is a scheduling request."
    prompt = f"""
    Determine if the following input is a scheduling request.
    Example:
    Input: Schedule a meeting for John and Jane.
    Output: schedule
    Input: What is the weather today?
    Output: weather
    Input: {question}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error analyzing input: {str(e)}"

def schedule_meeting(question):
    """Schedules a meeting using LLM and returns the proposed schedule with embedded example."""
    system_instruction = "You are an expert meeting scheduler."
    prompt = f"""
    Given the question, propose a meeting schedule.
    Example:
    Question: Schedule a meeting for John and Jane for 30 minutes on Monday. John is busy 1-2pm. Jane prefers to meet before noon.
    Output: Here is the proposed time: Monday, 9:00 - 9:30
    Question: {question}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error scheduling meeting: {str(e)}"

def verify_solution(problem, proposed_solution):
    """Verifies if the proposed solution is valid using a dedicated verification agent with embedded example."""
    system_instruction = "You are a strict meeting schedule validator. Verify if the proposed solution is valid."
    prompt = f"""
    Problem: {problem}
    Proposed Solution: {proposed_solution}
    Determine if the proposed solution is valid given the problem.
    Example:
    Problem: Schedule a meeting for John and Jane for 30 minutes on Monday. John is busy 1-2pm. Jane prefers to meet before noon. Proposed Solution: Monday, 1:30 - 2:00
    Output: Invalid - John is busy.

    Problem: Schedule a meeting for John and Jane for 30 minutes on Monday. John is busy 1-2pm. Jane prefers to meet before noon. Proposed Solution: Monday, 9:00 - 9:30
    Output: Valid.
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error verifying solution: {str(e)}"

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

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