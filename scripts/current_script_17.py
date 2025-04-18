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

def extract_meeting_data(question):
    """Extract meeting duration, participants, and raw schedule information using LLM with example."""
    system_instruction = "You are an expert at extracting data from scheduling questions."
    prompt = f"""
    Extract the meeting duration, participants, and the raw schedule information from the question.

    Example:
    Question: You need to schedule a meeting for Michelle, Andrea and Douglas for one hour between the work hours of 9:00 to 17:00 on Monday. 
    Michelle has blocked their calendar on Monday during 11:00 to 12:00, 14:00 to 15:00; 
    Andrea has meetings on Monday during 9:00 to 9:30, 11:30 to 12:30, 13:30 to 14:00, 14:30 to 15:00, 16:00 to 16:30; 
    Douglas has meetings on Monday during 9:00 to 9:30, 10:00 to 10:30, 11:00 to 15:00, 16:00 to 17:00; 
    
    Extracted Data:
    {{
        "duration": "1 hour",
        "participants": ["Michelle", "Andrea", "Douglas"],
        "raw_schedules": "Michelle has blocked their calendar on Monday during 11:00 to 12:00, 14:00 to 15:00; Andrea has meetings on Monday during 9:00 to 9:30, 11:30 to 12:30, 13:30 to 14:00, 14:30 to 15:00, 16:00 to 16:30; Douglas has meetings on Monday during 9:00 to 9:30, 10:00 to 10:30, 11:00 to 15:00, 16:00 to 17:00;"
    }}

    Question: {question}
    Extracted Data:
    """
    return call_llm(prompt, system_instruction)

def convert_schedules_to_time_slots(raw_schedules):
    """Convert raw schedules to structured time slots using LLM with example."""
    system_instruction = "You are an expert at converting free-form text schedules to structured time slots."
    prompt = f"""
    Convert the raw schedules text into a structured list of busy time slots for each participant. Assume the day is Monday.

    Example:
    Raw Schedules: Michelle has blocked their calendar on Monday during 11:00 to 12:00, 14:00 to 15:00; Andrea has meetings on Monday during 9:00 to 9:30, 11:30 to 12:30, 13:30 to 14:00, 14:30 to 15:00, 16:00 to 16:30; Douglas has meetings on Monday during 9:00 to 9:30, 10:00 to 10:30, 11:00 to 15:00, 16:00 to 17:00;
    
    Time Slots:
    {{
        "Michelle": ["11:00-12:00", "14:00-15:00"],
        "Andrea": ["9:00-9:30", "11:30-12:30", "13:30-14:00", "14:30-15:00", "16:00-16:30"],
        "Douglas": ["9:00-9:30", "10:00-10:30", "11:00-15:00", "16:00-17:00"]
    }}

    Raw Schedules: {raw_schedules}
    Time Slots:
    """
    return call_llm(prompt, system_instruction)

def find_available_time(participants, busy_slots, duration_hours):
    """Find an available meeting time given the participants, busy slots, and duration using LLM with example."""
    system_instruction = "You are an expert at finding an available time."
    prompt = f"""
    Given the participants, their busy time slots on Monday, and the required meeting duration, find an available time for the meeting on Monday.
    Assume the meeting must be scheduled between 9:00 and 17:00. Return 'No available time' if no solution exists.

    Example:
    Participants: ["Michelle", "Andrea", "Douglas"]
    Busy Slots:
    {{
        "Michelle": ["11:00-12:00", "14:00-15:00"],
        "Andrea": ["9:00-9:30", "11:30-12:30", "13:30-14:00", "14:30-15:00", "16:00-16:30"],
        "Douglas": ["9:00-9:30", "10:00-10:30", "11:00-15:00", "16:00-17:00"]
    }}
    Duration: 1 hour
    
    Reasoning:
    Michelle is free from 9:00-11:00, 12:00-14:00, 15:00-17:00
    Andrea is free from 9:30-11:30, 12:30-13:30, 14:00-14:30, 15:00-16:00
    Douglas is free from 9:30-10:00, 10:30-11:00, 15:00-16:00
    The only time where all participants are free is 15:00-16:00

    Available Time: Monday, 15:00 - 16:00

    Participants: {participants}
    Busy Slots: {busy_slots}
    Duration: {duration_hours}
    Reasoning:
    """
    return call_llm(prompt, system_instruction)

def is_valid_meeting_time(participants, busy_slots, proposed_time):
    """Check if the proposed meeting time is valid given the participants and busy slots using LLM with example."""
    system_instruction = "You are an expert at verifying if a proposed meeting time is valid given the participants and their busy slots."
    prompt = f"""
    Given the participants, their busy slots on Monday, and a proposed meeting time, verify if the proposed time is valid.

    Example:
    Participants: ["Michelle", "Andrea", "Douglas"]
    Busy Slots:
    {{
        "Michelle": ["11:00-12:00", "14:00-15:00"],
        "Andrea": ["9:00-9:30", "11:30-12:30", "13:30-14:00", "14:30-15:00", "16:00-16:30"],
        "Douglas": ["9:00-9:30", "10:00-10:30", "11:00-15:00", "16:00-17:00"]
    }}
    Proposed Time: Monday, 15:00 - 16:00
    
    Reasoning:
    Michelle is free from 15:00 to 16:00
    Andrea is free from 15:00 to 16:00
    Douglas is free from 15:00 to 16:00
    
    Verification: VALID

    Participants: {participants}
    Busy Slots: {busy_slots}
    Proposed Time: {proposed_time}
    Reasoning:
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule meetings."""
    try:
        # 1. Extract meeting data
        meeting_data = extract_meeting_data(question)
        meeting_data = json.loads(meeting_data)

        duration = meeting_data["duration"]
        participants = meeting_data["participants"]
        raw_schedules = meeting_data["raw_schedules"]

        # 2. Convert schedules to time slots
        busy_slots = convert_schedules_to_time_slots(raw_schedules)
        busy_slots = json.loads(busy_slots)

        # 3. Find available time
        available_time = find_available_time(participants, busy_slots, duration)

        # 4. Verify the solution
        is_valid = is_valid_meeting_time(participants, busy_slots, available_time)

        if "VALID" in is_valid:
            return f"Here is the proposed time: {available_time}"
        else:
            return "No suitable time slots found."

    except Exception as e:
        return f"Error: {str(e)}"