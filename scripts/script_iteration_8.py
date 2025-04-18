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
    """Schedules meetings using multi-stage extraction and iterative validation."""
    try:
        # 1. Extract meeting information - extract details individually
        participants = extract_participants(question)
        duration = extract_duration(question)
        days = extract_days(question)
        work_hours = extract_work_hours(question)
        schedules = extract_schedules(question, participants)

        # 2. Combine extracted information into a dictionary
        meeting_info = {
            "participants": participants,
            "duration": duration,
            "days": days,
            "work_hours": work_hours,
            "schedules": schedules
        }
        
        # 3. Find an available meeting slot
        available_slot = find_available_slot(meeting_info)
        
        return available_slot

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def extract_participants(question):
    """Extracts participants from the question using LLM."""
    system_instruction = "You are an expert at extracting participant names from meeting scheduling requests."
    prompt = f"""
    Extract the names of all participants from the following text.
    
    Example:
    Input: You need to schedule a meeting for John and Jane for 30 minutes between 9:00 and 17:00 on Monday.
    Output: ["John", "Jane"]
    
    Input: {question}
    Output:
    """
    try:
        extracted_info = call_llm(prompt, system_instruction)
        return json.loads(extracted_info)
    except Exception as e:
        return f"Error extracting info: {str(e)}"

def extract_duration(question):
    """Extracts the duration of the meeting from the question using LLM."""
    system_instruction = "You are an expert at extracting meeting duration from scheduling requests."
    prompt = f"""
    Extract the duration of the meeting (in minutes) from the following text.
    
    Example:
    Input: You need to schedule a meeting for John and Jane for 30 minutes between 9:00 and 17:00 on Monday.
    Output: 30
    
    Input: {question}
    Output:
    """
    try:
        extracted_info = call_llm(prompt, system_instruction)
        return int(extracted_info)
    except Exception as e:
        return f"Error extracting info: {str(e)}"

def extract_days(question):
    """Extracts the days the meeting can take place on from the question using LLM."""
    system_instruction = "You are an expert at extracting days from meeting scheduling requests."
    prompt = f"""
    Extract the days on which the meeting can take place from the following text.
    
    Example:
    Input: You need to schedule a meeting for John and Jane for 30 minutes between 9:00 and 17:00 on Monday.
    Output: ["Monday"]
    
    Input: {question}
    Output:
    """
    try:
        extracted_info = call_llm(prompt, system_instruction)
        return json.loads(extracted_info)
    except Exception as e:
        return f"Error extracting info: {str(e)}"

def extract_work_hours(question):
    """Extracts the work hours from the question using LLM."""
    system_instruction = "You are an expert at extracting work hours from meeting scheduling requests."
    prompt = f"""
    Extract the work hours (start and end time) from the following text.
    
    Example:
    Input: You need to schedule a meeting for John and Jane for 30 minutes between 9:00 and 17:00 on Monday.
    Output: ["9:00", "17:00"]
    
    Input: {question}
    Output:
    """
    try:
        extracted_info = call_llm(prompt, system_instruction)
        return json.loads(extracted_info)
    except Exception as e:
        return f"Error extracting info: {str(e)}"

def extract_schedules(question, participants):
    """Extracts the schedules of all participants from the question using LLM."""
    system_instruction = "You are an expert at extracting participant schedules from meeting scheduling requests."
    schedules = {}
    for participant in participants:
        prompt = f"""
        Extract the schedule of {participant} from the following text. Return the schedule as a list of time ranges.
        
        Example:
        Input: You need to schedule a meeting for John and Jane for 30 minutes between 9:00 and 17:00 on Monday. John is busy 10:00-11:00.
        Participant: John
        Output: [["10:00", "11:00"]]
        
        Input: {question}
        Participant: {participant}
        Output:
        """
        try:
            extracted_info = call_llm(prompt, system_instruction)
            schedules[participant] = json.loads(extracted_info)
        except Exception as e:
            schedules[participant] = []  # If fails, set empty schedule

    return schedules

def find_available_slot(meeting_info, max_attempts=5):
    """Finds available slots and validates them iteratively with LLM."""
    system_instruction = "You are an expert meeting scheduler."

    for attempt in range(max_attempts):
        # 1. Propose a meeting slot
        proposal_prompt = f"""
        Based on this meeting information: {meeting_info}, propose a possible meeting slot (day, start time - end time). Be mindful of work hours and participant schedules.

        Example:
        Meeting Info: {{"participants": ["John", "Jane"], "duration": 30, "days": ["Monday"], "work_hours": ["9:00", "17:00"], "schedules": {{"John": [["10:00", "11:00"]], "Jane": [["13:00", "14:00"]]}}}}
        Proposed Slot: Monday, 9:00 - 9:30

        Meeting Info: {meeting_info}
        Proposed Slot:
        """

        proposed_slot = call_llm(proposal_prompt, system_instruction)

        # 2. Validate proposed slot against constraints with dedicated validator
        validator_prompt = f"""
        You are an expert validator. Validate that this meeting slot: {proposed_slot} works for everyone and satisfies these meeting requirements: {meeting_info}.

        Example:
        Proposed Slot: Monday, 9:00 - 9:30
        Meeting Info: {{"participants": ["John", "Jane"], "duration": 30, "days": ["Monday"], "work_hours": ["9:00", "17:00"], "schedules": {{"John": [["10:00", "11:00"]], "Jane": [["13:00", "14:00"]]}}}}
        Validation: Valid

        Proposed Slot: {proposed_slot}
        Meeting Info: {meeting_info}
        Validation:
        """
        validator_system_instruction = "You are an expert validator that must validate meeting slots"
        validation_result = call_llm(validator_prompt, validator_system_instruction)

        if "Valid" in validation_result:
            return f"Here is the proposed time: {proposed_slot}"
        else:
            continue  # Retry with a new proposal

    return "Error: Could not find a suitable meeting time after multiple attempts."