import os
import json
import re

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

def extract_meeting_constraints(question):
    """Extract meeting constraints from the input question using LLM with examples."""
    system_instruction = "You are an expert at extracting meeting constraints from text."
    prompt = f"""
    Extract the following constraints from the question: participants, duration, days, start_time, end_time, existing schedules, preferences.

    Example:
    Question: You need to schedule a meeting for Carol and Mark for half an hour between the work hours of 9:00 to 17:00 on Monday. Carol has blocked their calendar on Monday during 10:00 to 11:00; Mark has blocked their calendar on Monday during 9:30 to 10:00.
    Let's think step by step. The participants are Carol and Mark. The duration is half an hour. The day is Monday. The start and end times are 9:00 and 17:00. Carol is busy from 10:00-11:00 and Mark is busy from 9:30-10:00.

    Extracted Constraints: {{"participants": ["Carol", "Mark"], "duration": "half an hour", "days": ["Monday"], "start_time": "9:00", "end_time": "17:00", "existing_schedules": {{"Carol": {{"Monday": ["10:00-11:00"]}}, "Mark": {{"Monday": ["9:30-10:00"]}}}}, "preferences": {{}}}}

    Question: {question}
    """
    return call_llm(prompt, system_instruction)

def propose_meeting_time(constraints_json):
    """Propose a meeting time using LLM reasoning with examples."""
    system_instruction = "You are an expert at proposing meeting times given constraints."
    prompt = f"""
    Given these meeting constraints, propose a meeting time. Explain your reasoning.

    Example:
    Constraints: {{"participants": ["Carol", "Mark"], "duration": "half an hour", "days": ["Monday"], "start_time": "9:00", "end_time": "17:00", "existing_schedules": {{"Carol": {{"Monday": ["10:00-11:00"]}}, "Mark": {{"Monday": ["9:30-10:00"]}}}}, "preferences": {{}}}}
    Let's think step by step. Carol is busy 10:00-11:00, and Mark is busy 9:30-10:00. A half-hour meeting needs to fit within 9:00-17:00. Therefore, 9:00-9:30 works.

    Proposed Time: Here is the proposed time: Monday, 9:00 - 9:30

    Constraints: {constraints_json}
    """
    return call_llm(prompt, system_instruction)

def verify_solution(question, proposed_time):
    """Verify if the proposed solution satisfies all requirements using LLM with examples."""
    system_instruction = "You are a critical evaluator who verifies meeting schedules."
    prompt = f"""
    Verify if the proposed meeting time satisfies all requirements in the question. List each constraint and verify it's met.

    Example:
    Question: You need to schedule a meeting for Carol and Mark for half an hour between 9:00 to 17:00 on Monday. Carol has blocked their calendar on Monday during 10:00 to 11:00; Mark has blocked their calendar on Monday during 9:30 to 10:00.
    Proposed Time: Monday, 9:00 - 9:30
    Let's think step by step.
    Constraint 1: Meeting is for Carol and Mark. Met.
    Constraint 2: Meeting is half an hour. Met.
    Constraint 3: Meeting is between 9:00 and 17:00 on Monday. Met.
    Constraint 4: Carol is not busy. Met.
    Constraint 5: Mark is not busy. Met.
    Verification: The proposed time satisfies all requirements.

    Question: {question}
    Proposed Time: {proposed_time}
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule a meeting."""
    try:
        # Extract meeting constraints
        constraints_json = extract_meeting_constraints(question)
        print(f"Extracted constraints: {constraints_json}") # Added logging
        # Propose a meeting time
        proposed_time = propose_meeting_time(constraints_json)
        print(f"Proposed time: {proposed_time}") # Added logging

        # Verify the solution
        verification_result = verify_solution(question, proposed_time)
        print(f"Verification result: {verification_result}") # Added logging

        return proposed_time
    except Exception as e:
        return f"Error: {str(e)}"