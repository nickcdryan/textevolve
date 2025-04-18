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
    """Orchestrates meeting scheduling using iterative extraction and verification."""
    # Step 1: Extract initial meeting details
    meeting_details = extract_meeting_details(question)

    # Step 2: Validate the extracted details and refine if necessary
    validated_details = validate_meeting_details(question, meeting_details)

    # Step 3: Generate candidate meeting times
    candidate_times = generate_candidate_times(validated_details)

    # Step 4: Check constraints against the candidate times
    available_times = check_constraints(validated_details, candidate_times)
    
    # Step 5: Return available times
    if available_times:
        solution = f"Here is the proposed time: {available_times[0]}"
    else:
        solution = "No suitable meeting time found."
    return solution

def extract_meeting_details(question):
    """Extracts meeting details from the input question."""
    system_instruction = "You are a meeting scheduling assistant."
    prompt = f"""
    Extract the following meeting details from the question: participants, duration, working hours, possible days, and existing schedules.

    Example Input:
    Schedule a meeting for John, Jane, and Peter for 30 minutes between 9:00 and 17:00 on Monday. John is busy from 10:00-11:00, Jane is busy from 14:00-15:00, and Peter is free all day.

    Expected Output:
    {{
      "participants": ["John", "Jane", "Peter"],
      "duration": "30 minutes",
      "working_hours": ["9:00", "17:00"],
      "possible_days": ["Monday"],
      "schedules": {{
        "John": {{"Monday": ["10:00-11:00"]}},
        "Jane": {{"Monday": ["14:00-15:00"]}},
        "Peter": {{"Monday": []}}
      }}
    }}

    Input Question: {question}
    """
    return call_llm(prompt, system_instruction)

def validate_meeting_details(question, meeting_details):
    """Validates the extracted meeting details and refines if necessary."""
    system_instruction = "You are a meeting scheduling expert."
    prompt = f"""
    Validate the following meeting details extracted from the question. If any details are incorrect or missing, correct them.

    Example Input:
    Question: Schedule a meeting for Alice and Bob for 1 hour between 9:00 and 17:00 on Tuesday. Alice is busy from 10:00-11:00.
    Extracted Details:
    {{
      "participants": ["Alice", "Bob"],
      "duration": "1 hour",
      "working_hours": ["9:00", "17:00"],
      "possible_days": ["Tuesday"],
      "schedules": {{
        "Alice": {{"Tuesday": ["10:00-11:00"]}},
        "Bob": {{"Tuesday": []}}
      }}
    }}
    
    Reasoning:
    The extracted details seem correct based on the question provided.
    
    Validated Details:
    {{
      "participants": ["Alice", "Bob"],
      "duration": "1 hour",
      "working_hours": ["9:00", "17:00"],
      "possible_days": ["Tuesday"],
      "schedules": {{
        "Alice": {{"Tuesday": ["10:00-11:00"]}},
        "Bob": {{"Tuesday": []}}
      }}
    }}

    Question: {question}
    Extracted Details: {meeting_details}
    """
    return call_llm(prompt, system_instruction)

def generate_candidate_times(validated_details):
    """Generates candidate meeting times."""
    working_hours = validated_details.split('"working_hours": [')[1].split(']')[0].replace('"', '').split(', ')
    start_time = working_hours[0]
    end_time = working_hours[1]
    duration = validated_details.split('"duration": "')[1].split('"')[0]
    duration_minutes = int(duration.split(" ")[0])

    possible_times = []
    days = validated_details.split('"possible_days": [')[1].split(']')[0].replace('"', '').split('", "')
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

def check_constraints(validated_details, candidate_times):
    """Checks constraints against the candidate meeting times."""
    schedule = validated_details.split('"schedules": {')[1].split('}')[0]
    available_times = []
    for time in candidate_times:
        day = time.split(",")[0]
        start_time = time.split(", ")[1].split(" - ")[0]
        end_time = time.split(" - ")[1]
        is_available = True

        participants = validated_details.split('"participants": [')[1].split(']')[0].replace('"', '').split('", "')
        for participant in participants:
            if participant in schedule:
                schedule_data = validated_details.split('"schedules": {')[1].split('}')[0]
                schedule_data_part = schedule_data.split(participant + '": {')[1].split('}')[0]
                busy_slots = []
                if day in schedule_data_part:
                    busy_slots = schedule_data_part.split('"'+day+'": [')[1].split(']')[0].replace('"', '').split('", "')

                for busy_slot in busy_slots:
                    if busy_slot != '':
                        busy_start, busy_end = busy_slot.split('-')
                        if not (end_time <= busy_start or start_time >= busy_end):
                            is_available = False
                            break
            if not is_available:
                break

        if is_available:
            available_times.append(time)
    return available_times