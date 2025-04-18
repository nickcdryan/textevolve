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
    Question: Schedule a meeting for John, Jane, and Mike.
    Participants: ["John", "Jane", "Mike"]

    Question: Schedule a meeting for Charles, Kayla, Cynthia, Rebecca, Randy and Hannah
    Participants: ["Charles", "Kayla", "Cynthia", "Rebecca", "Randy", "Hannah"]

    Question: {question}
    Participants:
    """
    return call_llm(prompt, system_instruction)

def extract_constraints(question):
    """Extract meeting constraints from the question using LLM."""
    system_instruction = "You are an expert at extracting scheduling constraints. Return as a plain text."
    prompt = f"""
    Extract the meeting constraints from the question, including unavailable times and preferred days.

    Example:
    Question: Schedule a meeting, John is busy Monday 9-10, Jane prefers Tuesdays.
    Constraints: John is busy Monday 9-10, Jane prefers Tuesdays.

    Question: Schedule a meeting for Charles, Kayla, Cynthia, Rebecca, Randy and Hannah for half an hour between the work hours of 9:00 to 17:00 on Monday. Kayla has meetings on Monday during 12:00 to 13:00; Randy is busy on Monday during 10:00 to 11:30, 12:00 to 13:30, 14:30 to 15:30, 16:00 to 17:00; Kayla do not want to meet on Monday before 10:30.
    Constraints: Kayla has meetings on Monday during 12:00 to 13:00; Randy is busy on Monday during 10:00 to 11:30, 12:00 to 13:30, 14:30 to 15:30, 16:00 to 17:00; Kayla do not want to meet on Monday before 10:30.

    Question: {question}
    Constraints:
    """
    return call_llm(prompt, system_instruction)

def solve_meeting_problem(participants, constraints, max_attempts=3):
    """Solve the meeting scheduling problem using LLM."""
    system_instruction = "You are an expert at solving meeting scheduling problems with constraints. Provide the result in a single sentence."
    prompt = f"""
    Given the participants and constraints, find a suitable meeting time. Consider all constraints and propose a valid time slot.

    Example:
    Participants: ["John", "Jane"]
    Constraints: John is busy Monday 9-10, Jane prefers Tuesdays.
    Solution: Monday, 11:00 - 11:30

    Participants: ["Charles", "Kayla", "Cynthia", Rebecca, Randy, Hannah]
    Constraints: Kayla has meetings on Monday during 12:00 to 13:00; Randy is busy on Monday during 10:00 to 11:30, 12:00 to 13:30, 14:30 to 15:30, 16:00 to 17:00; Kayla do not want to meet on Monday before 10:30.
    Solution: Monday, 14:00 - 14:30

    Participants: {participants}
    Constraints: {constraints}
    Solution:
    """
    return call_llm(prompt, system_instruction)

def verify_solution(question, solution):
    """Verify the proposed solution using LLM."""
    system_instruction = "You are an expert at verifying if a meeting time is valid. Provide the result VALID or INVALID."
    prompt = f"""
    Verify if the proposed meeting time is valid given the original question. Return just VALID or INVALID.

    Example:
    Question: Schedule a meeting for John, Jane, and Mike. John is busy Monday 9-10.
    Proposed Solution: Monday, 11:00 - 11:30
    Verification: VALID

    Question: Schedule a meeting for Charles, Kayla, Cynthia, Rebecca, Randy and Hannah for half an hour between the work hours of 9:00 to 17:00 on Monday. Kayla has meetings on Monday during 12:00 to 13:00; Randy is busy on Monday during 10:00 to 11:30, 12:00 to 13:30, 14:30 to 15:30, 16:00 to 17:00; Kayla do not want to meet on Monday before 10:30.
    Proposed Solution: Monday, 14:00 - 14:30
    Verification: VALID

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
            return "No suitable time slots found."

    except Exception as e:
        return f"Error: {str(e)}"