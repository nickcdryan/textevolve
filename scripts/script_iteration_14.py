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
    """Extract participant names from the question using LLM."""
    system_instruction = "You are an expert at extracting participant names."
    prompt = f"""
    Extract a list of participant names from the question.

    Example:
    Question: Schedule a meeting for John, Jane, and Mike for one hour.
    Participants: ["John", "Jane", "Mike"]

    Question: {question}
    Participants:
    """
    return call_llm(prompt, system_instruction)

def extract_constraints(question):
    """Extract meeting constraints from the question using LLM."""
    system_instruction = "You are an expert at extracting scheduling constraints."
    prompt = f"""
    Extract the meeting constraints from the question, including unavailable times, preferred days, and durations.

    Example:
    Question: Schedule a meeting, John is busy Monday 9-10, Jane prefers Tuesdays for one hour.
    Constraints: John is busy Monday 9-10, Jane prefers Tuesdays, Duration: 1 hour.

    Question: {question}
    Constraints:
    """
    return call_llm(prompt, system_instruction)

def solve_meeting_problem(participants, constraints):
    """Solve the meeting scheduling problem using LLM, considering earliest availability."""
    system_instruction = "You are an expert at solving meeting scheduling problems with constraints, always prioritizing earliest availability."
    prompt = f"""
    Given the participants and constraints, find the *earliest* suitable meeting time.

    Example:
    Participants: ["John", "Jane"]
    Constraints: John is busy Monday 9-10, Jane prefers Tuesdays.
    Solution: Tuesday, 11:00 - 11:30 (Earliest available time)

    Participants: {participants}
    Constraints: {constraints}
    Solution:
    """
    return call_llm(prompt, system_instruction)

def verify_solution(question, solution):
    """Verify the proposed solution using LLM."""
    system_instruction = "You are an expert at verifying if a meeting time is valid, considering all constraints and earliest availability."
    prompt = f"""
    Verify if the proposed meeting time is valid given the original question, and if it is the *earliest* possible time. If invalid, explain why.

    Example:
    Question: Schedule a meeting for John, Jane, and Mike. John is busy Monday 9-10.
    Proposed Solution: Monday, 11:00 - 11:30
    Verification: VALID - This time works for everyone and is the earliest available.

    Question: {question}
    Proposed Solution: {solution}
    Verification:
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule meetings."""
    try:
        # 1. Extract participants
        participants = extract_participants(question)

        # 2. Extract constraints
        constraints = extract_constraints(question)

        # 3. Solve the meeting problem
        solution = solve_meeting_problem(participants, constraints)

        # 4. Verify solution
        verification = verify_solution(question, solution)

        if "VALID" in verification:
            return f"Here is the proposed time: {solution}"
        else:
            return f"No suitable time slots found. Verification feedback: {verification}" #Includes the reason from the verifier

    except Exception as e:
        return f"Error: {str(e)}"