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

def extract_meeting_data(question):
    """Extract participants, duration, and working hours using LLM with examples."""
    system_instruction = "You are an expert in extracting data from scheduling requests."
    prompt = f"""
    Extract participant names, meeting duration, and working hours from the question.

    Example:
    Question: Schedule a meeting for John, Jane, and Mike for half an hour between 9:00 to 17:00 on Monday.
    Data:
    {{
      "participants": ["John", "Jane", "Mike"],
      "duration": "30 minutes",
      "working_hours": "9:00 to 17:00",
      "days": ["Monday"]
    }}

    Question: Schedule a meeting for Alice and Bob for 1 hour between 10:00 and 16:00 on Tuesday and Wednesday.
    Data:
    {{
      "participants": ["Alice", "Bob"],
      "duration": "1 hour",
      "working_hours": "10:00 to 16:00",
      "days": ["Tuesday", "Wednesday"]
    }}

    Question: {question}
    Data:
    """
    return call_llm(prompt, system_instruction)

def extract_availabilities(question, participants):
    """Extract availabilities of participants using LLM with examples."""
    system_instruction = "You are an expert in extracting availabilities of people."
    prompt = f"""
    For each participant, extract their availabilities (or busy times) from the question.

    Example:
    Question: Schedule a meeting for John and Jane. John is busy Monday 9-10. Jane is busy Tuesdays.
    John: John is busy Monday 9-10.
    Jane: Jane is busy Tuesdays.

    Question: Schedule a meeting for Charles, Kayla, Cynthia. Charles is free all day. Kayla is busy Monday 12-1.
    Charles: Charles is free all day.
    Kayla: Kayla is busy Monday 12-1.
    Cynthia: Cynthia is free all day.

    Participants: {participants}
    Question: {question}
    """
    return call_llm(prompt, system_instruction)

def find_free_slots(availabilities, working_hours, duration, days):
    """Use LLM to generate potential time slots with example."""
    system_instruction = "You are an expert at generating meeting schedules."
    prompt = f"""
    Given the availabilities, working hours, meeting duration and the desired days, determine the possible time slots.
    Availabilities: {availabilities}
    Working Hours: {working_hours}
    Duration: {duration}
    Days: {days}

    Example:
    Availabilities: John is busy Monday 9-10. Jane is busy Tuesdays.
    Working Hours: 9:00 to 17:00
    Duration: 30 minutes
    Days: Monday, Tuesday
    Possible Time Slots:
    Monday: 10:00-10:30, 10:30-11:00, 11:00-11:30, 11:30-12:00, 12:00-12:30, 12:30-13:00, 13:00-13:30, 13:30-14:00, 14:00-14:30, 14:30-15:00, 15:00-15:30, 15:30-16:00, 16:00-16:30, 16:30-17:00
    Tuesday: 9:00-9:30, 9:30-10:00, 10:00-10:30, 10:30-11:00, 11:00-11:30, 11:30-12:00, 12:00-12:30, 12:30-13:00, 13:00-13:30, 13:30-14:00, 14:00-14:30, 14:30-15:00, 15:00-15:30, 15:30-16:00, 16:00-16:30, 16:30-17:00

    Availabilities: {availabilities}
    Working Hours: {working_hours}
    Duration: {duration}
    Days: {days}
    Possible Time Slots:
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule meetings."""
    try:
        # Extract participants, duration, and working hours
        meeting_data = extract_meeting_data(question)
        meeting_data_json = json.loads(meeting_data)
        participants = meeting_data_json["participants"]
        duration = meeting_data_json["duration"]
        working_hours = meeting_data_json["working_hours"]
        days = meeting_data_json["days"]

        # Extract availabilities of each participant
        availabilities = extract_availabilities(question, participants)

        # Determine possible time slots
        possible_time_slots = find_free_slots(availabilities, working_hours, duration, days)

        return f"Here is the proposed time: {possible_time_slots}"

    except Exception as e:
        return f"Error: {str(e)}"