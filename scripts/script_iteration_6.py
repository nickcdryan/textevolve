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
    """Extract constraints and schedules from the question using LLM, aiming for plain-text output with improved examples."""
    system_instruction = "You are an expert at extracting meeting constraints in a structured way."

    prompt = f"""
    Extract the participants, duration, time constraints, existing schedules, and preferences from the given text.
    Return the information in a plain text format, with each piece of information on a new line.
    Explicitly mention if a participant has no meetings.

    Example 1:
    Text: You need to schedule a meeting for John and Jennifer for half an hour between 9:00 and 17:00 on Monday. John has no meetings. Jennifer has meetings on Monday from 9:00-11:00.
    Extraction:
    Participants: John, Jennifer
    Duration: 30 minutes
    Time Constraints: between 9:00 and 17:00 on Monday
    Existing Schedules: John - None, Jennifer - Monday 9:00-11:00

    Example 2:
    Text: You need to schedule a meeting for Patricia and Harold for half an hour on Monday. Patricia is busy 11:30-12:00. Harold is busy 9:30-10:30 and 11:30-12:00. Harold prefers not to meet after 14:00.
    Extraction:
    Participants: Patricia, Harold
    Duration: 30 minutes
    Time Constraints: Monday
    Existing Schedules: Patricia - Monday 11:30-12:00, Harold - Monday 9:30-10:30, 11:30-12:00
    Preferences: Harold - not after 14:00

    Example 3:
    Text: You need to schedule a meeting for Alice, Bob, and Carol for 45 minutes on Tuesday. Alice is available all day. Bob has a meeting from 10:00 to 11:00. Carol would prefer to meet before noon.
    Extraction:
    Participants: Alice, Bob, Carol
    Duration: 45 minutes
    Time Constraints: Tuesday
    Existing Schedules: Alice - None, Bob - Tuesday 10:00-11:00, Carol - None
    Preferences: Carol - before noon

    Text: {question}
    Extraction:
    """

    constraints = call_llm(prompt, system_instruction)
    return constraints

def schedule_meeting(constraints, max_attempts=3):
    """Schedule the meeting based on the extracted constraints using LLM with improved examples and explicit time slot generation."""
    system_instruction = "You are an expert at scheduling meetings, finding the best possible time while respecting preferences."

    prompt = f"""
    Given the following constraints, find a suitable time slot and respond with the complete sentence. Prioritize earlier times and respect all preferences. Show reasoning.

    Example 1:
    Constraints:
    Participants: John, Jennifer
    Duration: 30 minutes
    Time Constraints: between 9:00 and 17:00 on Monday
    Existing Schedules: John - None, Jennifer - Monday 9:00-11:00
    Reasoning: John is free all day. Jennifer is busy 9:00-11:00, so the earliest available time is 11:00. Proposing 11:00 to 11:30 to avoid conflict.
    Proposed Time: Here is the proposed time: Monday, 11:00 - 11:30

    Example 2:
    Constraints:
    Participants: Patricia, Harold
    Duration: 30 minutes
    Time Constraints: Monday
    Existing Schedules: Patricia - Monday 11:30-12:00, Harold - Monday 9:30-10:30, 11:30-12:00
    Preferences: Harold - not after 14:00
    Reasoning: Patricia is busy 11:30-12:00 and Harold is busy 9:30-10:30, 11:30-12:00. Harold prefers not to meet after 14:00. Considering the constraints, proposing 13:30 to 14:00.
    Proposed Time: Here is the proposed time: Monday, 13:30 - 14:00

    Example 3:
    Constraints:
    Participants: Alice, Bob, Carol
    Duration: 45 minutes
    Time Constraints: Tuesday
    Existing Schedules: Alice - None, Bob - Tuesday 10:00-11:00, Carol - None
    Preferences: Carol - before noon
    Reasoning: Alice and Carol are free all day. Bob is busy 10:00 to 11:00. Carol would prefer to meet before noon. Proposing a time of 9:00 - 9:45 to fulfill Carol's preference and avoid Bob's meeting
    Proposed Time: Here is the proposed time: Tuesday, 9:00 - 9:45

    Constraints:
    {constraints}
    Reasoning:
    Proposed Time:
    """

    schedule = call_llm(prompt, system_instruction)
    return schedule

def verify_schedule(constraints, schedule, max_attempts=3):
    """Verify that the proposed schedule satisfies all constraints using LLM with better examples and detailed INVALID reasoning."""
    system_instruction = "You are a meeting schedule verifier. Provide detailed reasoning for INVALID schedules."

    prompt = f"""
    You are given extracted constraints and a proposed meeting schedule. Determine if the schedule satisfies all constraints.
    Respond with VALID if the schedule is valid, or INVALID: [reason] if it is not. Provide specific reasons for the verdict.

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
    Verification: INVALID: Harold prefers not to meet after 14:00. The proposed time violates Harold's preference.

    Example 3:
    Constraints:
    Participants: Alice, Bob, Carol
    Duration: 45 minutes
    Time Constraints: Tuesday
    Existing Schedules: Alice - None, Bob - Tuesday 10:00-11:00, Carol - None
    Preferences: Carol - before noon
    Proposed Schedule: Here is the proposed time: Tuesday, 10:15 - 11:00
    Verification: INVALID: Bob is busy from 10:00-11:00 on Tuesday, so 10:15-11:00 doesn't work.

    Constraints:
    {constraints}
    Proposed Schedule: {schedule}
    Verification:
    """

    verification = call_llm(prompt, system_instruction)
    return verification

def main(question):
    """Main function to process the question and return the answer with robust error handling."""
    try:
        constraints = extract_constraints(question)
        if "Error" in constraints:  # Handle errors during constraint extraction
            return f"Constraint Extraction Error: {constraints}"

        schedule = schedule_meeting(constraints)
        if "Error" in schedule:  # Handle errors during schedule generation
            return f"Schedule Generation Error: {schedule}"

        verification = verify_schedule(constraints, schedule)

        if "INVALID" in verification:
            return f"Error: {verification}" #Return verification error if invalid

        return schedule
    except Exception as e:
        return f"Error: {str(e)}" #Catch all exceptions