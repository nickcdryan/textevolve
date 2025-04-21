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

def extract_constraints(question, max_attempts=3):
    """Extract constraints and schedules from the question using LLM, aiming for plain-text output."""
    system_instruction = "You are an expert at extracting meeting constraints in a structured way."

    prompt = f"""
    Extract the participants, duration, time constraints, and existing schedules from the given text.
    Return the information in a plain text format, with each piece of information on a new line.

    Example:
    Text: You need to schedule a meeting for John and Jennifer for half an hour between 9:00 and 17:00 on Monday. John has no meetings. Jennifer has meetings on Monday from 9:00-11:00.
    Extraction:
    Participants: John, Jennifer
    Duration: 30 minutes
    Time Constraints: between 9:00 and 17:00 on Monday
    Existing Schedules: John - None, Jennifer - Monday 9:00-11:00

    Text: You need to schedule a meeting for Patricia and Harold for half an hour on Monday. Patricia is busy 11:30-12:00. Harold is busy 9:30-10:30 and 11:30-12:00. Harold prefers not to meet after 14:00.
    Extraction:
    Participants: Patricia, Harold
    Duration: 30 minutes
    Time Constraints: Monday
    Existing Schedules: Patricia - Monday 11:30-12:00, Harold - Monday 9:30-10:30, 11:30-12:00
    Preferences: Harold - not after 14:00

    Text: {question}
    Extraction:
    """

    constraints = call_llm(prompt, system_instruction)
    return constraints

def schedule_meeting(constraints, max_attempts=3):
    """Schedule the meeting based on the extracted constraints using LLM."""
    system_instruction = "You are an expert at scheduling meetings, finding the best possible time."

    prompt = f"""
    Given the following constraints, find a suitable time slot and respond with the complete sentence.

    Example 1:
    Constraints:
    Participants: John, Jennifer
    Duration: 30 minutes
    Time Constraints: between 9:00 and 17:00 on Monday
    Existing Schedules: John - None, Jennifer - Monday 9:00-11:00
    Proposed Time: Here is the proposed time: Monday, 13:00 - 13:30

    Example 2:
    Constraints:
    Participants: Patricia, Harold
    Duration: 30 minutes
    Time Constraints: Monday
    Existing Schedules: Patricia - Monday 11:30-12:00, Harold - Monday 9:30-10:30, 11:30-12:00
    Preferences: Harold - not after 14:00
    Proposed Time: Here is the proposed time: Monday, 13:30 - 14:00

    Constraints:
    {constraints}
    Proposed Time:
    """

    schedule = call_llm(prompt, system_instruction)
    return schedule

def verify_schedule(constraints, schedule, max_attempts=3):
    """Verify that the proposed schedule satisfies all constraints using LLM."""
    system_instruction = "You are a meeting schedule verifier."

    prompt = f"""
    You are given extracted constraints and a proposed meeting schedule. Determine if the schedule satisfies all constraints.
    Respond with VALID if the schedule is valid, or INVALID: [reason] if it is not.

    Example 1:
    Constraints:
    Participants: John, Jennifer
    Duration: 30 minutes
    Time Constraints: between 9:00 and 17:00 on Monday
    Existing Schedules: John - None, Jennifer - Monday 9:00-11:00
    Proposed Schedule: Here is the proposed time: Monday, 13:00 - 13:30
    Verification: VALID

    Example 2:
    Constraints:
    Participants: Patricia, Harold
    Duration: 30 minutes
    Time Constraints: Monday
    Existing Schedules: Patricia - Monday 11:30-12:00, Harold - Monday 9:30-10:30, 11:30-12:00
    Preferences: Harold - not after 14:00
    Proposed Schedule: Here is the proposed time: Monday, 15:00 - 15:30
    Verification: INVALID: Harold prefers not to meet after 14:00

    Constraints:
    {constraints}
    Proposed Schedule: {schedule}
    Verification:
    """

    verification = call_llm(prompt, system_instruction)
    return verification

def main(question):
    """Main function to process the question and return the answer."""
    try:
        constraints = extract_constraints(question)
        schedule = schedule_meeting(constraints)
        verification = verify_schedule(constraints, schedule)

        if "INVALID" in verification:
            return f"Error: {verification}"

        return schedule
    except Exception as e:
        return f"Error: {str(e)}"