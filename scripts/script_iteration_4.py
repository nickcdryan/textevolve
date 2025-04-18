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

def extract_participants(question, max_attempts=3):
    """Extract participant names from the question using LLM with retries and examples."""
    system_instruction = "You are an expert at extracting participant names from scheduling requests."
    prompt = f"""
    Extract a list of participant names from the question. Return a JSON list.

    Example:
    Question: Schedule a meeting for John, Jane, and Mike.
    Participants: ["John", "Jane", "Mike"]

    Question: {question}
    Participants:
    """
    for attempt in range(max_attempts):
        try:
            participants_str = call_llm(prompt, system_instruction)
            participants = json.loads(participants_str)
            return participants
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Attempt {attempt + 1} failed to parse participants: {e}")
            if attempt == max_attempts - 1:
                return [] # Return an empty list on final failure
            # Adjust the prompt on retry, adding more structure
            prompt += "\nEnsure your response is a valid JSON list."
    return []

def extract_constraints(question, max_attempts=3):
    """Extract meeting constraints from the question using LLM with retries and examples."""
    system_instruction = "You are an expert at extracting scheduling constraints from meeting requests."
    prompt = f"""
    Extract the meeting constraints from the question, including unavailable times and preferred days.
    Return the constraints as a string.

    Example:
    Question: Schedule a meeting, John is busy Monday 9-10, Jane prefers Tuesdays.
    Constraints: John is busy Monday 9-10, Jane prefers Tuesdays.

    Question: {question}
    Constraints:
    """
    for attempt in range(max_attempts):
        try:
            constraints = call_llm(prompt, system_instruction)
            return constraints
        except Exception as e:
            print(f"Attempt {attempt + 1} failed to extract constraints: {e}")
            if attempt == max_attempts - 1:
                return "" # Return an empty string on final failure
            prompt += "\nProvide the constraints as a simple, readable string."
    return ""

def solve_meeting_problem(participants, constraints, max_attempts=3):
    """Solve the meeting scheduling problem using LLM with improved examples."""
    system_instruction = "You are an expert at solving meeting scheduling problems with complex constraints. Provide the solution as a string."
    prompt = f"""
    Given the participants and constraints, find a suitable meeting time.

    Example:
    Participants: ["John", "Jane"]
    Constraints: John is busy Monday 9-10, Jane prefers Tuesdays.
    Solution: Tuesday, 11:00 - 11:30

    Participants: {participants}
    Constraints: {constraints}
    Solution:
    """
    for attempt in range(max_attempts):
        try:
            solution = call_llm(prompt, system_instruction)
            return solution
        except Exception as e:
            print(f"Attempt {attempt + 1} failed to solve meeting problem: {e}")
            if attempt == max_attempts - 1:
                return "No suitable time slots found." # Return default on final failure
            prompt += "\nProvide the solution as a day and time range, like 'Monday, 14:00 - 14:30'."
    return "No suitable time slots found."

def verify_solution(question, solution, max_attempts=3):
    """Verify the proposed solution using LLM with clear examples."""
    system_instruction = "You are an expert at verifying if a proposed meeting time is valid. Respond with 'VALID' or 'INVALID'."
    prompt = f"""
    Verify if the proposed meeting time is valid given the original question. Respond ONLY with 'VALID' or 'INVALID'.

    Example:
    Question: Schedule a meeting for John, Jane, and Mike. John is busy Monday 9-10.
    Proposed Solution: Monday, 11:00 - 11:30
    Verification: VALID

    Question: {question}
    Proposed Solution: {solution}
    Verification:
    """
    for attempt in range(max_attempts):
        try:
            verification = call_llm(prompt, system_instruction)
            if "VALID" in verification.upper():
                return "VALID"
            else:
                return "INVALID"
        except Exception as e:
            print(f"Attempt {attempt + 1} failed to verify solution: {e}")
            if attempt == max_attempts - 1:
                return "INVALID"
            prompt += "\nRespond ONLY with the word 'VALID' or 'INVALID'."
    return "INVALID"

def main(question):
    """Main function to schedule meetings."""
    try:
        # 1. Extract participants
        participants = extract_participants(question)
        if not participants:
            return "Error: Could not extract participants."

        # 2. Extract constraints
        constraints = extract_constraints(question)
        if not constraints:
            return "Error: Could not extract constraints."

        # 3. Solve the meeting problem
        solution = solve_meeting_problem(participants, constraints)
        if "No suitable time slots found" in solution:
            return "No suitable time slots found."

        # 4. Verify solution
        verification = verify_solution(question, solution)

        if verification == "VALID":
            return f"Here is the proposed time: {solution}"
        else:
            return "No suitable time slots found."

    except Exception as e:
        return f"Error: {str(e)}"