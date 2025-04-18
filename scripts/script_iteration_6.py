import os
import json
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
    """Extract participant names from the question using LLM with an example."""
    system_instruction = "You are an expert at extracting participant names from scheduling requests."
    prompt = f"""
    Extract the participant names from the question.

    Example:
    Question: Schedule a meeting for John, Jane, and Mike.
    Participants: John, Jane, Mike

    Question: {question}
    Participants:
    """
    try:
        participants = call_llm(prompt, system_instruction)
        return [p.strip() for p in participants.split(',')]  # Split and strip names
    except Exception as e:
        print(f"Error extracting participants: {e}")
        return []  # Return empty list on error

def extract_constraints_with_example(question):
    """Extract scheduling constraints using LLM with an example and structured output."""
    system_instruction = "You are an expert at extracting scheduling constraints from meeting requests, providing structured output."
    prompt = f"""
    Extract the scheduling constraints from the question and format them as a list of sentences.

    Example:
    Question: John is busy Monday 9-10, Jane prefers Tuesdays. Schedule a meeting for them.
    Constraints:
    - John is busy Monday 9-10.
    - Jane prefers Tuesdays.

    Question: {question}
    Constraints:
    """
    try:
        constraints = call_llm(prompt, system_instruction)
        return [c.strip() for c in constraints.split('\n') if c.strip()]  # Split and strip constraints
    except Exception as e:
        print(f"Error extracting constraints: {e}")
        return []  # Return empty list on error

def solve_meeting_problem(participants, constraints):
    """Solve the scheduling problem using LLM with example."""
    system_instruction = "You are an expert at solving meeting scheduling problems with constraints."
    prompt = f"""
    Given the participants and constraints, find a suitable meeting time.

    Example:
    Participants: John, Jane
    Constraints:
    - John is busy Monday 9-10.
    - Jane prefers Tuesdays.
    Solution: Tuesday, 11:00 - 11:30

    Participants: {', '.join(participants)}
    Constraints:
    {chr(10).join(constraints)}
    Solution:
    """
    try:
        solution = call_llm(prompt, system_instruction)
        return solution
    except Exception as e:
        print(f"Error solving the meeting problem: {e}")
        return "No suitable time slots found."

def main(question):
    """Main function to schedule meetings."""
    try:
        # 1. Extract participants
        participants = extract_participants(question)
        if not participants:
            return "Error: Could not extract participants."

        # 2. Extract constraints
        constraints = extract_constraints_with_example(question)
        if not constraints:
            return "Error: Could not extract constraints."

        # 3. Solve the meeting problem
        solution = solve_meeting_problem(participants, constraints)

        return f"Here is the proposed time: {solution}"

    except Exception as e:
        return f"Error: {str(e)}"