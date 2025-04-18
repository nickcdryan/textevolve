import re
import json
import datetime
from datetime import timedelta

def main(question):
    """
    Schedules meetings by extracting info, generating slots, filtering, and selecting the best slot.
    Uses LLM reasoning for info extraction and preference handling, and deterministic logic for slot generation.
    """
    meeting_info = extract_meeting_info(question)
    possible_slots = generate_meeting_slots(meeting_info)
    filtered_slots = filter_meeting_slots(meeting_info, possible_slots)
    best_slot = select_best_meeting_slot(meeting_info, filtered_slots)

    return best_slot

def extract_meeting_info(question):
    """Extracts meeting details (participants, duration, days, work hours, existing schedules, preferences) using LLM."""
    system_instruction = "You are an expert meeting scheduler. Extract all relevant information."
    prompt = f"""
    Extract the following information from the question:
    - participants: List of people involved in the meeting.
    - duration: Meeting length in minutes.
    - days: List of days the meeting can occur (e.g., Monday, Tuesday).
    - work_hours: Start and end times for work hours (e.g., 9:00 to 17:00).
    - existing_schedules: A dictionary of schedules for each participant, with blocked time ranges.
    - preferences: Any preferences of participants like avoiding meetings after a certain time.

    Example:
    Question: You need to schedule a meeting for John and Jennifer for half an hour between the work hours of 9:00 to 17:00 on either Monday, Tuesday or Wednesday. 
    Here are the existing schedules for everyone during the days: 
    Johnhas no meetings the whole week.
    Jennifer has meetings on Monday during 9:00 to 11:00, 11:30 to 13:00, 13:30 to 14:30, 15:00 to 17:00, Tuesday during 9:00 to 11:30, 12:00 to 17:00, Wednesday during 9:00 to 11:30, 12:00 to 12:30, 13:00 to 14:00, 14:30 to 16:00, 16:30 to 17:00; 
    John would like to avoid more meetings on Monday after 14:30. Tuesday. Wednesday. Find a time that works for everyone's schedule and constraints.
    Output:
    {{
      "participants": ["John", "Jennifer"],
      "duration": 30,
      "days": ["Monday", "Tuesday", "Wednesday"],
      "work_hours": ["9:00", "17:00"],
      "existing_schedules": {{
        "John": {{
          "Monday": [],
          "Tuesday": [],
          "Wednesday": []
        }},
        "Jennifer": {{
          "Monday": [["9:00", "11:00"], ["11:30", "13:00"], ["13:30", "14:30"], ["15:00", "17:00"]],
          "Tuesday": [["9:00", "11:30"], ["12:00", "17:00"]],
          "Wednesday": [["9:00", "11:30"], ["12:00", "12:30"], ["13:00", "14:00"], ["14:30", "16:00"], ["16:30", "17:00"]]
        }}
      }},
      "preferences": {{
        "John": {{"Monday": "14:30"}}
      }}
    }}
    Question: {question}
    """

    try:
        llm_response = call_llm(prompt, system_instruction)
        meeting_info = json.loads(llm_response)

        # Convert time strings to datetime.time objects, and duration to integer
        for person, schedule in meeting_info["existing_schedules"].items():
            for day, blocked_times in schedule.items():
                for i, (start, end) in enumerate(blocked_times):
                    meeting_info["existing_schedules"][person][day][i] = [parse_time(start), parse_time(end)]
        meeting_info["duration"] = int(meeting_info["duration"])

        return meeting_info
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error extracting meeting info: {e}")
        return None

def generate_meeting_slots(meeting_info):
    """Generates all possible meeting slots within constraints, using deterministic logic."""
    slots = []
    start_time = parse_time(meeting_info["work_hours"][0])
    end_time = parse_time(meeting_info["work_hours"][1])
    duration = meeting_info["duration"]
    for day in meeting_info["days"]:
        current_time = start_time
        while current_time + timedelta(minutes=duration) <= end_time:
            slots.append({"day": day, "start": current_time, "end": current_time + timedelta(minutes=duration)})
            current_time += timedelta(minutes=15)  # Increment by 15 minutes to find more slots
    return slots

def filter_meeting_slots(meeting_info, possible_slots):
    """Filters out invalid meeting slots based on participants' existing schedules."""
    filtered_slots = []
    for slot in possible_slots:
        valid = True
        for person, schedule in meeting_info["existing_schedules"].items():
            for blocked_time in schedule.get(slot["day"], []):
                if slot["start"] < blocked_time[1] and slot["end"] > blocked_time[0]:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            filtered_slots.append(slot)
    return filtered_slots

def select_best_meeting_slot(meeting_info, filtered_slots):
    """Selects the best meeting slot based on participant preferences, using LLM."""
    system_instruction = "You are an expert at selecting the best meeting slot based on preferences."
    prompt = f"""
    Given these possible meeting slots: {filtered_slots}, and the participants' preferences: {meeting_info["preferences"]}, select the best slot based on the preferences.
    If there are no preferences, return the first available slot.

    Example:
    Meeting slots: [{{'day': 'Monday', 'start': datetime.time(13, 0), 'end': datetime.time(13, 30)}}, {{'day': 'Monday', 'start': datetime.time(14, 0), 'end': datetime.time(14, 30)}}]
    Preferences: {{'John': {{'Monday': '14:30'}}}}
    Output: Here is the proposed time: Monday, 13:00 - 13:30

    Meeting slots: {filtered_slots}
    Preferences: {meeting_info["preferences"]}
    """
    try:
        if not filtered_slots:
            return "No available slots found."
        llm_response = call_llm(prompt, system_instruction)
        return "Here is the proposed time: " + llm_response
    except Exception as e:
        print(f"Error selecting best slot: {e}")
        return "Error selecting best slot."

def parse_time(time_str):
    """Parses a time string (e.g., "9:00") and returns a datetime.time object."""
    try:
        return datetime.datetime.strptime(time_str, "%H:%M").time()
    except ValueError:
        return None

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response"""
    import os
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