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

def extract_meeting_data(question, max_attempts=3):
    """
    Extract structured data from the meeting scheduling question.
    Uses the LLM to produce a simplified, non-JSON output for reliability.
    """
    system_instruction = "You are an expert at extracting meeting details and providing structured outputs."

    prompt = f"""
    Extract structured data about the meeting from the following text.
    Output participants, duration, constraints, schedules, and preferences, each on a new line.

    Example:
    Text: Schedule a meeting for John and Jennifer for half an hour between 9:00 and 17:00 on Monday. John has no meetings. Jennifer has meetings on Monday from 9:00-11:00.
    Output:
    Participants: John, Jennifer
    Duration: 30 minutes
    Constraints: 9:00 to 17:00, Monday
    Schedules: John - None, Jennifer - Monday 9:00-11:00
    Preferences: None

    Text: Schedule a meeting for Patricia and Harold for half an hour on Monday. Patricia is busy 11:30-12:00. Harold is busy 9:30-10:30 and 11:30-12:00. Harold prefers not to meet after 14:00.
    Output:
    Participants: Patricia, Harold
    Duration: 30 minutes
    Constraints: Monday
    Schedules: Patricia - Monday 11:30-12:00, Harold - Monday 9:30-10:30, 11:30-12:00
    Preferences: Harold - not after 14:00

    Text: {question}
    Output:
    """

    extracted_data = call_llm(prompt, system_instruction)
    return extracted_data

def schedule_meeting(meeting_data, max_attempts=3):
    """
    Schedule a meeting using the structured data extracted from the question.
    """
    system_instruction = "You are an expert at scheduling meetings given structured data."

    prompt = f"""
    Given the following structured data, schedule the meeting.

    Example:
    Data:
    Participants: John, Jennifer
    Duration: 30 minutes
    Constraints: 9:00 to 17:00, Monday
    Schedules: John - None, Jennifer - Monday 9:00-11:00
    Preferences: None
    Proposed Schedule: Here is the proposed time: Monday, 13:00 - 13:30

    Data:
    Participants: Patricia, Harold
    Duration: 30 minutes
    Constraints: Monday
    Schedules: Patricia - Monday 11:30-12:00, Harold - Monday 9:30-10:30, 11:30-12:00
    Preferences: Harold - not after 14:00
    Proposed Schedule: Here is the proposed time: Monday, 13:30 - 14:00

    Data: {meeting_data}
    Proposed Schedule:
    """

    proposed_schedule = call_llm(prompt, system_instruction)
    return proposed_schedule

def verify_solution(question, extracted_data, proposed_schedule, max_attempts=3):
    """
    Verify the solution with the LLM verifier.
    """
    system_instruction = "You are a highly skilled verifier."

    prompt = f"""
    You are given a meeting scheduling question, extracted data from the question and a proposed schedule. Determine if the schedule is valid based on the data.

    Example:
    Question: Schedule a meeting for John and Jennifer for half an hour between 9:00 and 17:00 on Monday. John has no meetings. Jennifer has meetings on Monday from 9:00-11:00.
    Extracted Data:
    Participants: John, Jennifer
    Duration: 30 minutes
    Constraints: 9:00 to 17:00, Monday
    Schedules: John - None, Jennifer - Monday 9:00-11:00
    Preferences: None
    Proposed Schedule: Here is the proposed time: Monday, 13:00 - 13:30
    Assessment: VALID

    Question: Schedule a meeting for Patricia and Harold for half an hour on Monday. Patricia is busy 11:30-12:00. Harold is busy 9:30-10:30 and 11:30-12:00. Harold prefers not to meet after 14:00.
    Extracted Data:
    Participants: Patricia, Harold
    Duration: 30 minutes
    Constraints: Monday
    Schedules: Patricia - Monday 11:30-12:00, Harold - Monday 9:30-10:30, 11:30-12:00
    Preferences: Harold - not after 14:00
    Proposed Schedule: Here is the proposed time: Monday, 15:00 - 15:30
    Assessment: INVALID - Harold prefers not to meet after 14:00.

    Question: {question}
    Extracted Data: {extracted_data}
    Proposed Schedule: {proposed_schedule}
    Assessment:
    """

    verification_result = call_llm(prompt, system_instruction)
    return verification_result

def main(question):
    """
    Main function to schedule a meeting.
    """
    try:
        extracted_data = extract_meeting_data(question)
        proposed_schedule = schedule_meeting(extracted_data)
        verification_result = verify_solution(question, extracted_data, proposed_schedule)

        if "INVALID" in verification_result:
            return f"Error: {verification_result}"
        else:
            return proposed_schedule
    except Exception as e:
        return f"Error: {str(e)}"