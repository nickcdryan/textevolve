import json
import os
import re
import math

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response"""
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

def extract_participants(question):
    """Extract participant names from the question using LLM with example."""
    system_instruction = "You are an expert at extracting participant names."
    prompt = f"""
    Extract a list of participant names from the question.

    Example:
    Question: Schedule a meeting for John, Jane, and Mike.
    Reasoning: The participants are John, Jane, and Mike.
    Participants: ["John", "Jane", "Mike"]

    Question: {question}
    Reasoning:
    """
    return call_llm(prompt, system_instruction)

def extract_constraints(question):
    """Extract meeting constraints from the question using LLM with example."""
    system_instruction = "You are an expert at extracting scheduling constraints."
    prompt = f"""
    Extract the meeting constraints from the question, including unavailable times and preferred days.

    Example:
    Question: Schedule a meeting, John is busy Monday 9-10, Jane prefers Tuesdays.
    Reasoning: John cannot attend Monday 9-10, and Jane prefers to attend on Tuesdays.
    Constraints: John is busy Monday 9-10, Jane prefers Tuesdays.

    Question: {question}
    Reasoning:
    """
    return call_llm(prompt, system_instruction)

def solve_meeting_problem(participants, constraints):
    """Solve the meeting scheduling problem using LLM with example."""
    system_instruction = "You are an expert at solving meeting scheduling problems with constraints."
    prompt = f"""
    Given the participants and constraints, find a suitable meeting time.
    Ensure the time works for all participants.

    Example:
    Participants: ["John", "Jane"]
    Constraints: John is busy Monday 9-10, Jane prefers Tuesdays.
    Reasoning: Since John cannot attend Monday 9-10 and Jane prefers Tuesday, propose a time on Tuesday.
    Solution: Tuesday, 11:00 - 11:30

    Participants: {participants}
    Constraints: {constraints}
    Reasoning:
    """
    return call_llm(prompt, system_instruction)

def verify_solution(question, solution):
    """Verify the proposed solution using LLM with example."""
    system_instruction = "You are an expert at verifying if a meeting time is valid."
    prompt = f"""
    Verify if the proposed meeting time is valid given the original question.

    Example:
    Question: Schedule a meeting for John, Jane, and Mike. John is busy Monday 9-10.
    Proposed Solution: Monday, 11:00 - 11:30
    Reasoning: The proposed time does not conflict with John being busy on Monday 9-10
    Verification: VALID

    Question: {question}
    Proposed Solution: {solution}
    Reasoning:
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule meetings."""
    try:
        # 1. Extract participants
        participants = extract_participants(question)
        print(f"Participants: {participants}") # Debug

        # 2. Extract constraints
        constraints = extract_constraints(question)
        print(f"Constraints: {constraints}") # Debug

        # 3. Solve the meeting problem
        solution = solve_meeting_problem(participants, constraints)
        print(f"Solution: {solution}") # Debug

        # 4. Verify solution
        verification = verify_solution(question, solution)
        print(f"Verification: {verification}") # Debug

        if "VALID" in verification:
            return f"Here is the proposed time: {solution}"
        else:
            return "No suitable time slots found."

    except Exception as e:
        return f"Error: {str(e)}"