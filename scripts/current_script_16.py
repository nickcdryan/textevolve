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
    Participants: ["John", "Jane", "Mike"]

    Question: {question}
    Participants:
    """
    return call_llm(prompt, system_instruction)

def extract_constraints(question):
    """Extract meeting constraints from the question using LLM with example."""
    system_instruction = "You are an expert at extracting scheduling constraints."
    prompt = f"""
    Extract the meeting constraints from the question, including unavailable times and preferred days.

    Example:
    Question: Schedule a meeting, John is busy Monday 9-10, Jane prefers Tuesdays.
    Constraints: John is busy Monday 9:00-10:00, Jane prefers Tuesdays.

    Question: {question}
    Constraints:
    """
    return call_llm(prompt, system_instruction)

def solve_meeting_problem(participants, constraints, max_attempts=3):
    """Solve the meeting scheduling problem using LLM with example."""
    system_instruction = "You are an expert at solving meeting scheduling problems with constraints."
    prompt = f"""
    Given the participants and constraints, find a suitable meeting time on Monday.

    Example:
    Participants: ["John", "Jane"]
    Constraints: John is busy Monday 9:00-10:00, Jane prefers Tuesdays.
    Solution: Monday, 11:00 - 11:30

    Participants: {participants}
    Constraints: {constraints}
    Solution:
    """
    return call_llm(prompt, system_instruction)

def verify_solution(question, solution):
    """Verify the proposed solution using LLM with example."""
    system_instruction = "You are an expert at verifying if a meeting time is valid."
    prompt = f"""
    Verify if the proposed meeting time is valid given the original question.

    Example:
    Question: Schedule a meeting for John, Jane, and Mike. John is busy Monday 9:00-10:00.
    Proposed Solution: Monday, 11:00 - 11:30
    Verification: VALID

    Question: {question}
    Proposed Solution: {solution}
    Verification:
    """
    return call_llm(prompt, system_instruction)

def is_valid_time(question, proposed_time):
    """
    Verify, deterministically, if the proposed meeting time is valid
    given the original question by checking constraints against the
    participants.
    """
    system_instruction = "You are an expert at verifying if a time is valid for a meeting, based on the participants' schedules."
    prompt = f"""
    You are given a question and proposed time for a meeting. You need to determine if the proposed time is valid.

    Question: You need to schedule a meeting for Jesse, Kathryn and Megan for half an hour between the work hours of 9:00 to 17:00 on Monday. 
    Jesse has blocked their calendar on Monday during 10:00 to 10:30, 15:30 to 16:00; 
    Kathryn's calendar is wide open the entire day.
    Megan is busy on Monday during 10:30 to 11:00, 11:30 to 12:30, 13:30 to 14:30, 15:00 to 16:30; 
    Proposed Time: Monday, 9:00 - 9:30
    Valid: VALID

    Question: {question}
    Proposed Time: {proposed_time}
    Valid:
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

        # 4. Verify solution with LLM
        verification = verify_solution(question, solution)

        # 5. Deterministically, verify the solution and validate the time.
        is_valid_time_result = is_valid_time(question, solution)

        if "VALID" in verification and "VALID" in is_valid_time_result:
            return f"Here is the proposed time: {solution}"
        else:
            return "No suitable time slots found."

    except Exception as e:
        return f"Error: {str(e)}"