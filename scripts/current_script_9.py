import os
import json
import re
import datetime
from datetime import timedelta

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

def main(question):
    """Schedules meetings using iterative extraction and validation."""
    try:
        # 1. Extract key information (participants, duration, days, hours)
        extracted_info = extract_key_info(question)
        if "Error" in extracted_info:
            return "Error extracting key information."
        info = json.loads(extracted_info)

        # 2. Extract and validate schedules using iterative approach
        schedules = extract_and_validate_schedules(question, info['participants'])

        # 3. Find available slot based on extracted info and schedules
        available_slot = find_available_time_slot(info, schedules)

        return available_slot

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def extract_key_info(question):
    """Extracts key information (participants, duration, days, hours) with example."""
    system_instruction = "You are an expert at extracting key meeting details."
    prompt = f"""
    Extract the key information from the following meeting scheduling request and respond in JSON format.

    Example:
    Input: You need to schedule a meeting for John and Jane for 30 minutes between 9:00 and 17:00 on Monday.
    Output:
    {{
      "participants": ["John", "Jane"],
      "duration": 30,
      "days": ["Monday"],
      "work_hours": ["9:00", "17:00"]
    }}

    Now, extract the key info from the following:
    {question}
    """
    return call_llm(prompt, system_instruction)

def extract_and_validate_schedules(question, participants):
    """Extracts and validates schedules iteratively."""
    schedules = {}
    for participant in participants:
        schedule = extract_schedule(question, participant)
        schedules[participant] = schedule
    return schedules

def extract_schedule(question, participant):
    """Extracts schedule for a participant with verification loop."""
    system_instruction = "You are an expert at extracting meeting schedules."
    prompt = f"""
    Extract the schedule for {participant} from the following text: {question}.
    Return the schedule as a list of time ranges (e.g., [["10:00", "11:00"]]).
    If no schedule is mentioned, return an empty list.

    Example:
    Input: You need to schedule a meeting for John and Jane for 30 minutes. John is busy 10:00-11:00.
    Participant: John
    Output: [["10:00", "11:00"]]

    Input: {question}
    Participant: {participant}
    Output:
    """
    return json.loads(call_llm(prompt, system_instruction))

def find_available_time_slot(info, schedules):
    """Finds an available time slot considering all constraints."""
    system_instruction = "You are an expert meeting scheduler."
    prompt = f"""
    Given the following meeting information and schedules, find an available time slot.
    Meeting Information: {info}
    Schedules: {schedules}

    Consider the following constraints:
    - Participants must be available
    - Meeting duration must be respected
    - Work hours must be respected

    Example:
    Meeting Information: {{"participants": ["John", "Jane"], "duration": 30, "days": ["Monday"], "work_hours": ["9:00", "17:00"]}}
    Schedules: {{"John": [["10:00", "11:00"]], "Jane": [["13:00", "14:00"]]}}
    Output: Here is the proposed time: Monday, 9:00 - 9:30

    Find available time slot for the following:
    """
    combined_info = {"Meeting Information": info, "Schedules": schedules}
    prompt += json.dumps(combined_info)
    result = call_llm(prompt, system_instruction)
    return "Here is the proposed time: " + result