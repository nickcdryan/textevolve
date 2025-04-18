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

def extract_info_and_verify(question, max_attempts=3):
    """Extract meeting info with verification loop."""
    for attempt in range(max_attempts):
        info = extract_meeting_info(question)
        if info and verify_extracted_info(question, info):
            return info
    return None

def extract_meeting_info(question):
    """Extract meeting info from the question."""
    system_instruction = "You are a meeting scheduling expert."
    prompt = f"""
    Extract meeting details including participants, duration, work hours, and schedules.

    Example:
    Question: Schedule a meeting for John and Jane for 30 minutes between 9am-5pm. John: Mon 9-10am. Jane: Mon 10-11am.
    Extracted Info:
    {{
      "participants": ["John", "Jane"], "duration": 30, "work_hours": [9, 17],
      "schedules": {{"John": [["Mon", 9, 10]], "Jane": [["Mon", 10, 11]]}}
    }}

    Question: {question}
    """
    try:
        response = call_llm(prompt, system_instruction)
        return json.loads(response)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"JSON Error: {e}")
        return None

def verify_extracted_info(question, info):
    """Verify extracted information against the original question."""
    system_instruction = "You are a verification expert."
    prompt = f"""
    Verify if the extracted info matches the question.

    Example:
    Question: Schedule a meeting for John and Jane for 30 minutes between 9am-5pm. John: Mon 9-10am. Jane: Mon 10-11am.
    Extracted Info:
    {{
      "participants": ["John", "Jane"], "duration": 30, "work_hours": [9, 17],
      "schedules": {{"John": [["Mon", 9, 10]], "Jane": [["Mon", 10, 11]]}}
    }}
    Verification: True

    Question: {question}
    Extracted Info: {info}
    Verification:
    """
    response = call_llm(prompt, system_instruction)
    return "True" in response

def find_available_slots(extracted_info):
    """Find available time slots based on extracted information."""
    # Placeholder logic. In a real implementation, this function would calculate available time slots.
    return ["Monday 14:00 - 14:30"]

def filter_slots_by_constraints(extracted_info, time_slots):
    """Filter available time slots based on constraints."""
    system_instruction = "You are a constraint-based time slot filter."
    prompt = f"""
    Filter these time slots based on schedules:

    Example:
    Time Slots: ["Monday 9:00 - 9:30"]
    Schedules: {{"John": [["Mon", 9, 10]]}}
    Filtered Slots: []

    Time Slots: {time_slots}
    Schedules: {extracted_info.get("schedules", {{}})}
    Filtered Slots:
    """
    response = call_llm(prompt, system_instruction)
    return json.loads(response)

def main(question):
    """Main function to schedule meetings."""
    extracted_info = extract_info_and_verify(question)
    if not extracted_info:
        return "Error: Could not extract or verify meeting information."

    time_slots = find_available_slots(extracted_info)
    filtered_slots = filter_slots_by_constraints(extracted_info, time_slots)

    if filtered_slots:
        return f"Here is the proposed time: {filtered_slots[0]}"
    else:
        return "No suitable time slots found."