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

def extract_meeting_constraints(question):
    """
    Extract meeting constraints from the question, including participants, duration,
    available days, work hours, and individual preferences/schedules.
    Uses an LLM with embedded examples for robust extraction.
    """
    system_instruction = "You are an expert information extractor."
    prompt = f"""
    Extract the following information from the input text:
    - Participants: A list of people who need to attend the meeting.
    - Duration: The length of the meeting in minutes.
    - Available Days: The days of the week the meeting can be scheduled on.
    - Work Hours: The start and end time of the workday.
    - Individual Schedules: Each participant's busy times on the available days.
    - Preferences: Any specific time preferences or constraints mentioned.

    Example:
    Input: You need to schedule a meeting for Nicholas, Sara, Helen for half an hour between the work hours of 9:00 to 17:00 on Monday.
    Nicholas is busy on Monday during 9:00 to 9:30; Sara is busy on Monday during 10:00 to 10:30; Helen is free the entire day.

    Extracted Information:
    {{
        "participants": ["Nicholas", "Sara", "Helen"],
        "duration": 30,
        "available_days": ["Monday"],
        "work_hours": ["9:00", "17:00"],
        "individual_schedules": {{
            "Nicholas": [["9:00", "9:30"]],
            "Sara": [["10:00", "10:30"]],
            "Helen": []
        }},
        "preferences": []
    }}

    Now, extract the information from the following input:
    {question}
    """
    return call_llm(prompt, system_instruction)

def find_available_time_slots(constraints_json):
    """
    Find available time slots that satisfy the extracted meeting constraints.
    Uses an LLM with embedded examples for reasoning.
    """
    system_instruction = "You are an expert at finding available time slots."
    prompt = f"""
    Given the meeting constraints, find a suitable time slot for the meeting.
    Consider all participants' schedules, the meeting duration, and any preferences.

    Constraints:
    {constraints_json}

    Example:
    Constraints:
    {{
        "participants": ["Nicholas", "Sara", "Helen"],
        "duration": 30,
        "available_days": ["Monday"],
        "work_hours": ["9:00", "17:00"],
        "individual_schedules": {{
            "Nicholas": [["9:00", "9:30"]],
            "Sara": [["10:00", "10:30"]],
            "Helen": []
        }},
        "preferences": []
    }}

    Reasoning:
    1. All participants are required to attend the meeting.
    2. The meeting duration is 30 minutes.
    3. The meeting must be on Monday between 9:00 and 17:00.
    4. Nicholas is busy from 9:00 to 9:30, Sara is busy from 10:00 to 10:30, and Helen is free.
    5. Iterate through the available time slots and find a 30-minute slot that works for everyone.
    6. A possible time slot is 9:30 to 10:00. Nicholas is available, Sara is available, and Helen is available.

    Proposed Time:
    Monday, 9:30 - 10:00

    Now, find a suitable time slot for the given constraints:
    """
    return call_llm(prompt, system_instruction)

def verify_solution(constraints_json, proposed_time):
    """
    Verify if the proposed time slot satisfies all constraints using embedded examples.
    """
    system_instruction = "You are an expert at verifying if a proposed solution satisfies all the given constraints."
    prompt = f"""
    You are given a set of constraints and a proposed time slot. Your job is to verify if the proposed time slot
    satisfies all the constraints.

    Constraints:
    {constraints_json}

    Proposed Time:
    {proposed_time}

    Example:
    Constraints:
    {{
        "participants": ["Nicholas", "Sara", "Helen"],
        "duration": 30,
        "available_days": ["Monday"],
        "work_hours": ["9:00", "17:00"],
        "individual_schedules": {{
            "Nicholas": [["9:00", "9:30"]],
            "Sara": [["10:00", "10:30"]],
            "Helen": []
        }},
        "preferences": []
    }}
    Proposed Time:
    Monday, 9:30 - 10:00

    Reasoning:
    1. Check if all participants are available during the proposed time.
    2. Nicholas is available from 9:30 to 10:00.
    3. Sara is available from 9:30 to 10:00.
    4. Helen is available from 9:30 to 10:00.
    5. The proposed time is within the work hours of 9:00 to 17:00.
    6. The meeting duration is 30 minutes.
    7. All constraints are satisfied.

    Verification Result:
    The proposed time is valid.

    Now, verify if the proposed time satisfies the given constraints:
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """
    Main function to solve the meeting scheduling problem.
    """
    try:
        # Extract meeting constraints
        constraints_json = extract_meeting_constraints(question)

        # Find available time slots
        proposed_time = find_available_time_slots(constraints_json)

        # Verify the solution
        verification_result = verify_solution(constraints_json, proposed_time)

        return f"Here is the proposed time: {proposed_time}"
    except Exception as e:
        return f"Error: {str(e)}"