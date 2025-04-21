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

def extract_info_with_react(question, max_attempts=3):
    """Extract meeting information using a ReAct approach."""
    system_instruction = "You are an expert meeting scheduler that extracts details using the ReAct pattern."

    prompt = f"""
    You need to extract the participants, duration, constraints, schedules, and preferences from the following meeting scheduling question.
    Use the ReAct pattern to determine the best way to find the information. After extracting all the information, respond with a final thought, which will be a final extraction with the information that was extracted from the question.

    Example 1:
    Question: You need to schedule a meeting for John and Jennifer for half an hour between the work hours of 9:00 to 17:00 on Monday. John has no meetings. Jennifer has meetings on Monday from 9:00-11:00.
    Thought 1: I need to identify the participants in the meeting.
    Action 1: Search[participants]
    Observation 1: John and Jennifer
    Thought 2: I need to identify the duration of the meeting.
    Action 2: Search[duration]
    Observation 2: half an hour
    Thought 3: I need to identify the time constraints for the meeting.
    Action 3: Search[time constraints]
    Observation 3: between 9:00 to 17:00 on Monday
    Thought 4: I need to identify the schedules for the meeting.
    Action 4: Search[schedules]
    Observation 4: John has no meetings. Jennifer has meetings on Monday from 9:00-11:00.
    Thought 5: I need to identify any preferences.
    Action 5: Search[preferences]
    Observation 5: None
    Thought 6: I have identified all of the information for the meeting.
    Action 6: Finish[participants: John, Jennifer; duration: half an hour; constraints: between 9:00 to 17:00 on Monday; schedules: John - None, Jennifer - Monday 9:00-11:00; preferences: None]

    Question: {question}
    """

    extracted_info = call_llm(prompt, system_instruction)
    return extracted_info

def schedule_meeting(extracted_info, max_attempts=3):
    """Schedule the meeting based on the extracted information."""
    system_instruction = "You are an expert at scheduling meetings, generating the best answer possible."

    prompt = f"""
    Given the extracted meeting information, schedule the meeting.
    Return the result as a complete sentence starting with "Here is the proposed time:".

    Example 1:
    Extracted Information: participants: John, Jennifer; duration: half an hour; constraints: between 9:00 to 17:00 on Monday; schedules: John - None, Jennifer - Monday 9:00-11:00; preferences: None
    Proposed Time: Here is the proposed time: Monday, 13:00 - 13:30

    Extracted Information: {extracted_info}
    Proposed Time:
    """
    schedule = call_llm(prompt, system_instruction)
    return schedule

def verify_schedule(question, extracted_info, schedule, max_attempts=3):
    """Verify that the proposed schedule satisfies all constraints."""
    system_instruction = "You are a meeting schedule verifier, confirming if the schedule is valid."

    prompt = f"""
    You are given a meeting question, extracted information, and a proposed schedule. Determine if the schedule works with the information and constraints that were given from the question.
    If any constraints or requirements aren't met, respond with INVALID:[Reason for invalid schedule]. Otherwise, respond with VALID
    Example 1:
    Question: You need to schedule a meeting for John and Jennifer for half an hour between the work hours of 9:00 to 17:00 on Monday. John has no meetings. Jennifer has meetings on Monday from 9:00-11:00.
    Extracted Information: participants: John, Jennifer; duration: half an hour; constraints: between 9:00 to 17:00 on Monday; schedules: John - None, Jennifer - Monday 9:00-11:00; preferences: None
    Proposed Schedule: Here is the proposed time: Monday, 13:00 - 13:30
    Validation: VALID

    Example 2:
    Question: You need to schedule a meeting for John and Jennifer for half an hour between the work hours of 9:00 to 17:00 on Monday. John has no meetings. Jennifer has meetings on Monday from 9:00-11:00.
    Extracted Information: participants: John, Jennifer; duration: half an hour; constraints: between 9:00 to 17:00 on Monday; schedules: John - None, Jennifer - Monday 9:00-11:00; preferences: None
    Proposed Schedule: Here is the proposed time: Monday, 9:00 - 9:30
    Validation: INVALID: Jennifer has a meeting from 9:00 to 11:00

    Question: {question}
    Extracted Information: {extracted_info}
    Proposed Schedule: {schedule}
    Validation:
    """
    verification = call_llm(prompt, system_instruction)
    return verification

def main(question):
    """Main function to process the question and return the answer."""
    try:
        extracted_info = extract_info_with_react(question)
        schedule = schedule_meeting(extracted_info)
        verification = verify_schedule(question, extracted_info, schedule)

        if "INVALID" in verification:
            return f"Error: {verification}"

        return schedule
    except Exception as e:
        return f"Error: {str(e)}"