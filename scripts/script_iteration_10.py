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
    """Extract meeting constraints using LLM with embedded examples."""
    system_instruction = "You are an expert at extracting structured data from text."
    prompt = f"""
    Extract the following information from the provided text as a JSON object:
    - participants: A list of the participants in the meeting.
    - duration: The duration of the meeting in minutes.
    - available_days: A list of the days of the week the meeting can be scheduled on.
    - start_time: The earliest time the meeting can start (e.g., "9:00").
    - end_time: The latest time the meeting can end (e.g., "17:00").
    - schedules: A dictionary where keys are participant names and values are lists of tuples representing busy time slots (day, start_time, end_time).

    Example:
    Text: You need to schedule a meeting for Jack and Jill for 30 minutes between 9:00 and 17:00 on Monday, Tuesday, or Wednesday.
    Jack is busy on Monday from 10:00-11:00 and Tuesday from 14:00-15:00. Jill is busy on Wednesday from 9:30-10:00.
    Output:
    {{
        "participants": ["Jack", "Jill"],
        "duration": 30,
        "available_days": ["Monday", "Tuesday", "Wednesday"],
        "start_time": "9:00",
        "end_time": "17:00",
        "schedules": {{
            "Jack": [["Monday", "10:00", "11:00"], ["Tuesday", "14:00", "15:00"]],
            "Jill": [["Wednesday", "9:30", "10:00"]]
        }}
    }}

    Now extract information from this text:
    {question}
    """
    try:
        constraints_json = call_llm(prompt, system_instruction)
        constraints = json.loads(constraints_json)
        return constraints
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Error extracting constraints: {e}")
        return None
    except Exception as e:
        print(f"Error extracting constraints: {e}")
        return None

def find_available_time_slot(constraints):
    """Find an available time slot using extracted constraints and LLM for verification."""
    try:
        participants = constraints["participants"]
        duration = constraints["duration"]
        available_days = constraints["available_days"]
        start_time = constraints["start_time"]
        end_time = constraints["end_time"]
        schedules = constraints["schedules"]
    except (TypeError, KeyError) as e:
        print(f"Missing constraints: {e}")
        return "Error: Could not extract all required constraints."

    # First, generate possible time slots within the given constraints
    possible_slots = []
    for day in available_days:
        start_hour, start_minute = map(int, start_time.split(':'))
        end_hour, end_minute = map(int, end_time.split(':'))
        current_hour, current_minute = start_hour, start_minute
        while current_hour < end_hour or (current_hour == end_hour and current_minute <= end_minute - duration):
            slot_start = f"{current_hour:02}:{current_minute:02}"
            slot_end_minute = current_minute + duration
            slot_end_hour = current_hour
            if slot_end_minute >= 60:
                slot_end_minute -= 60
                slot_end_hour += 1
            slot_end = f"{slot_end_hour:02}:{slot_end_minute:02}"
            possible_slots.append((day, slot_start, slot_end))
            current_minute += 30
            if current_minute >= 60:
                current_minute -= 60
                current_hour += 1

    # Now, verify each time slot with LLM
    for day, slot_start, slot_end in possible_slots:
        available = True
        for participant in participants:
            if participant in schedules:
                for busy_day, busy_start, busy_end in schedules[participant]:
                    if day == busy_day:
                        # Check if the proposed slot overlaps with any busy slot
                        if not (slot_end <= busy_start or slot_start >= busy_end):
                            available = False
                            break
            if not available:
                break
        if available:
            # Verification step using LLM
            verification_result = verify_time_slot(day, slot_start, slot_end, participants, schedules)
            if "VALID" in verification_result:
                return f"Here is the proposed time: {day}, {slot_start} - {slot_end}"

    return "No available time slots found."

def verify_time_slot(day, slot_start, slot_end, participants, schedules):
    """Verify if the proposed time slot is valid using LLM with embedded examples."""
    system_instruction = "You are an expert at verifying time slots against schedules."
    prompt = f"""
    You are given a proposed time slot and a list of participant schedules. Determine if the time slot is valid, meaning that all participants are available during the entire time slot.

    Example:
    Day: Monday, Start Time: 11:00, End Time: 11:30
    Participants: ["Jack", "Jill"]
    Schedules:
    {{
        "Jack": [["Monday", "10:00", "11:00"], ["Tuesday", "14:00", "15:00"]],
        "Jill": [["Wednesday", "9:30", "10:00"]]
    }}
    Reasoning:
    - Jack is busy on Monday from 10:00 to 11:00, so he is available from 11:00 to 11:30.
    - Jill has no meetings on Monday, so she is available.
    Conclusion: VALID

    Now verify this time slot:
    Day: {day}, Start Time: {slot_start}, End Time: {slot_end}
    Participants: {participants}
    Schedules: {schedules}
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule a meeting."""
    constraints = extract_meeting_constraints(question)
    if constraints:
        answer = find_available_time_slot(constraints)
        return answer
    else:
        return "Error: Could not extract meeting details."