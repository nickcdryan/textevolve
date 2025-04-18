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

def main(question):
    """Main function to schedule meetings based on given constraints."""
    try:
        # 1. Extract information using a structured extraction prompt with examples
        extraction_result = extract_meeting_details(question)
        meeting_details = json.loads(extraction_result)

        # 2. Generate possible meeting times
        possible_times = generate_meeting_times(meeting_details)

        # 3. Filter possible times based on participant schedules
        filtered_times = filter_available_times(possible_times, meeting_details)

        # 4. Select the best meeting time (or indicate no solution)
        if filtered_times:
            solution = f"Here is the proposed time: {filtered_times[0]}"
        else:
            solution = "No suitable meeting time found."
        return solution

    except Exception as e:
        return f"Error: {str(e)}"

def extract_meeting_details(question):
    """Extract meeting details from the input text using LLM with examples."""
    system_instruction = "You are an expert information extractor, skilled at identifying relevant meeting details."

    prompt = f"""
    Extract the key details required to schedule a meeting from the input text. Identify participants, duration, working hours, possible days, and existing schedules.

    Example Input:
    You need to schedule a meeting for Joyce, Christine and Alexander for half an hour between the work hours of 9:00 to 17:00 on Monday.
    Joyce has meetings on Monday during 11:00 to 11:30, 13:30 to 14:00, 14:30 to 16:30;
    Christinehas no meetings the whole day.
    Alexander has meetings on Monday during 9:00 to 11:00, 12:00 to 12:30, 13:30 to 15:00, 15:30 to 16:00, 16:30 to 17:00;
    Christine can not meet on Monday before 12:00.

    Expected Output:
    {{
      "participants": ["Joyce", "Christine", "Alexander"],
      "duration": "30 minutes",
      "working_hours": ["9:00", "17:00"],
      "possible_days": ["Monday"],
      "schedules": {{
        "Joyce": {{"Monday": ["11:00-11:30", "13:30-14:00", "14:30-16:30"]}},
        "Christine": {{"Monday": []}},
        "Alexander": {{"Monday": ["9:00-11:00", "12:00-12:30", "13:30-15:00", "15:30-16:00", "16:30-17:00"]}}
      }},
      "constraints": {{"Christine": {{"Monday": "before 12:00"}}}}
    }}

    Input Text:
    {question}
    """
    return call_llm(prompt, system_instruction)

def generate_meeting_times(meeting_details):
    """Generate possible meeting times."""
    start_time = meeting_details["working_hours"][0]
    end_time = meeting_details["working_hours"][1]
    duration_minutes = int(meeting_details["duration"].split(" ")[0])

    possible_times = []
    days = meeting_details["possible_days"]
    for day in days:
        current_time = start_time
        while True:
            start_hour, start_minute = map(int, current_time.split(':'))
            end_hour, end_minute = map(int, end_time.split(':'))
            if start_hour > end_hour or (start_hour == end_hour and start_minute >= end_minute):
                break
            
            possible_end_minute = start_minute + duration_minutes
            possible_end_hour = start_hour
            if possible_end_minute >= 60:
                possible_end_hour = start_hour + 1
                possible_end_minute = possible_end_minute - 60
                
            #Format End Time
            end_time_string = str(possible_end_hour).zfill(2) + ":" + str(possible_end_minute).zfill(2)
            
            #Add in Zfill
            possible_times.append(f"{day}, {current_time} - {end_time_string}")
            
            start_minute = start_minute + 30
            if start_minute >= 60:
                start_hour = start_hour + 1
                start_minute = start_minute - 60

            current_time = str(start_hour).zfill(2) + ":" + str(start_minute).zfill(2)

    return possible_times

def filter_available_times(possible_times, meeting_details):
    """Filter possible times based on participant schedules and constraints."""
    available_times = []
    for time in possible_times:
        day = time.split(",")[0]
        start_time = time.split(", ")[1].split(" - ")[0]
        end_time = time.split(" - ")[1]
        is_available = True

        for participant, schedule_data in meeting_details["schedules"].items():
            schedule = schedule_data.get(day, [])
            for busy_slot in schedule:
                busy_start, busy_end = busy_slot.split('-')
                if not (end_time <= busy_start or start_time >= busy_end):
                    is_available = False
                    break
            if not is_available:
                break

        if is_available:
            available_times.append(time)
    return available_times