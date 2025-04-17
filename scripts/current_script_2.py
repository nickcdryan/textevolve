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
    Extracted Constraints: {{"participants": ["Carol", "Mark"], "duration": "half an hour", "days": ["Monday"], "start_time": "9:00", "end_time": "17:00", "existing_schedules": {{"Carol": {{"Monday": ["10:00-11:00"]}}, "Mark": {{"Monday": ["9:30-10:00"]}}}}, "preferences": {{}}}}

    Question: You need to schedule a meeting for John, Jane, and Mike for 1 hour between 9:00 and 17:00 on Tuesday. John is busy from 10:00-12:00, Jane from 13:00-14:00, and Mike has no conflicts. John prefers not to meet before 11:00.
    Extracted Constraints: {{"participants": ["John", "Jane", "Mike"], "duration": "1 hour", "days": ["Tuesday"], "start_time": "9:00", "end_time": "17:00", "existing_schedules": {{"John": {{"Tuesday": ["10:00-12:00"]}}, "Jane": {{"Tuesday": ["13:00-14:00"]}}, "Mike": {{"Tuesday": []}}}}, "preferences": {{"John": "not before 11:00"}}}}

    Question: {question}
    """
    return call_llm(prompt, system_instruction)

def propose_meeting_time(constraints_json):
    """Propose a meeting time using LLM reasoning with examples."""
    system_instruction = "You are an expert at proposing meeting times given constraints."
    prompt = f"""
    Given these meeting constraints, propose a meeting time.
    Example:
    Constraints: {{"participants": ["Carol", "Mark"], "duration": "half an hour", "days": ["Monday"], "start_time": "9:00", "end_time": "17:00", "existing_schedules": {{"Carol": {{"Monday": ["10:00-11:00"]}}, "Mark": {{"Monday": ["9:30-10:00"]}}}}, "preferences": {{}}}}
    Proposed Time: Here is the proposed time: Monday, 9:00 - 9:30

    Constraints: {{"participants": ["John", "Jane", "Mike"], "duration": "1 hour", "days": ["Tuesday"], "start_time": "9:00", "end_time": "17:00", "existing_schedules": {{"John": {{"Tuesday": ["10:00-12:00"]}}, "Jane": {{"Tuesday": ["13:00-14:00"]}}, "Mike": {{"Tuesday": []}}}}, "preferences": {{"John": "not before 11:00"}}}}
    Proposed Time: Here is the proposed time: Tuesday, 12:00 - 13:00

    Constraints: {constraints_json}
    """
    return call_llm(prompt, system_instruction)

def verify_solution(question, proposed_time):
    """Verify if the proposed solution satisfies all requirements using LLM with examples."""
    system_instruction = "You are a critical evaluator who verifies meeting schedules."
    prompt = f"""
    Verify if the proposed meeting time satisfies all requirements in the question.
    Example:
    Question: You need to schedule a meeting for Carol and Mark for half an hour between 9:00 to 17:00 on Monday. Carol has blocked their calendar on Monday during 10:00 to 11:00; Mark has blocked their calendar on Monday during 9:30 to 10:00.
    Proposed Time: Monday, 9:00 - 9:30
    Verification: The proposed time satisfies all requirements.

    Question: You need to schedule a meeting for John, Jane, and Mike for 1 hour between 9:00 and 17:00 on Tuesday. John is busy from 10:00-12:00, Jane from 13:00-14:00, and Mike has no conflicts. John prefers not to meet before 11:00.
    Proposed Time: Tuesday, 12:00 - 13:00
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

        # Propose a meeting time
        proposed_time = propose_meeting_time(constraints_json)

        # Verify the solution
        verification_result = verify_solution(question, proposed_time)

        return proposed_time
    except Exception as e:
        return f"Error: {str(e)}"