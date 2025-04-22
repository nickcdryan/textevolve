import os
import re
import math
import json

def main(question):
    """Schedules meetings by extracting info, identifying slots, proposing times, and verifying."""
    meeting_info = extract_meeting_info(question)
    available_slots = identify_available_time_slots(meeting_info)
    proposed_time = propose_meeting_time(available_slots, meeting_info)
    return verify_final_solution(proposed_time, meeting_info)

def extract_meeting_info(question):
    """Extracts meeting details (participants, duration, constraints) from the question."""
    system_instruction = "You are an expert at extracting meeting information from text."
    prompt = f"""
    Extract the following information from the text: participants, duration, work hours, days, existing schedules.
    
    Example 1:
    Text: You need to schedule a meeting for John and Jennifer for half an hour between the work hours of 9:00 to 17:00 on either Monday, Tuesday or Wednesday. 
    John has no meetings the whole week. Jennifer has meetings on Monday during 9:00 to 11:00.
    Output: {{"participants": ["John", "Jennifer"], "duration": "half an hour", "work_hours": "9:00 to 17:00", "days": ["Monday", "Tuesday", "Wednesday"], "schedules": {{"John": "John has no meetings the whole week.", "Jennifer": "Jennifer has meetings on Monday during 9:00 to 11:00."}}}}
    
    Example 2:
    Text: Schedule a meeting for Patricia and Harold for 45 minutes on Friday between 10:00 and 16:00. Patricia is free all day. Harold has a meeting from 11:00 to 12:00.
    Output: {{"participants": ["Patricia", "Harold"], "duration": "45 minutes", "work_hours": "10:00 to 16:00", "days": ["Friday"], "schedules": {{"Patricia": "Patricia is free all day.", "Harold": "Harold has a meeting from 11:00 to 12:00."}}}}
    
    Text: {question}
    Output:
    """
    return call_llm(prompt, system_instruction)

def identify_available_time_slots(meeting_info):
    """Identifies available time slots based on participants' schedules."""
    system_instruction = "You are an expert at identifying available time slots based on schedules."
    prompt = f"""
    Based on the meeting information and schedules, identify the available time slots.
    
    Example:
    Meeting Info: {{"participants": ["John", "Jennifer"], "duration": "half an hour", "work_hours": "9:00 to 17:00", "days": ["Monday"], "schedules": {{"John": "John has no meetings the whole week.", "Jennifer": "Jennifer has meetings on Monday during 9:00 to 11:00."}}}}
    Available Time Slots: John is free all day. Jennifer is available from 11:00 to 17:00.
    
   Meeting Info: {meeting_info}
   Available Time Slots:
    """
    return call_llm(prompt, system_instruction)

def propose_meeting_time(available_slots, meeting_info):
    """Proposes a specific meeting time based on available slots."""
    system_instruction = "You are an expert at proposing specific meeting times."
    prompt = f"""
    Based on the available time slots and meeting information, propose a specific meeting time.
    
    Example:
    Available Time Slots: John is free all day. Jennifer is available from 11:00 to 17:00. Meeting Duration: half an hour. Days: Monday.
    Proposed Time: Here is the proposed time: Monday, 13:00 - 13:30
    
    Available Time Slots: {available_slots}. Meeting Info: {meeting_info}
    Proposed Time: Here is the proposed time:
    """
    return call_llm(prompt, system_instruction)

def verify_final_solution(proposed_time, meeting_info):
    """Verifies if the proposed solution satisfies all requirements."""
    system_instruction = "You are a meticulous meeting scheduler. Verify if the proposed time works based on the schedule."
    prompt = f"""
    You are a verification agent who validates the proposed time to check if the suggested time works for all particpants and doesn't conflicts with their schedule. Use only the provided information to perform the verification.
    
    Example 1:
    Meeting Info: {{"participants": ["John", "Jennifer"], "duration": "half an hour", "work_hours": "9:00 to 17:00", "days": ["Monday", "Tuesday", "Wednesday"], "schedules": {{"John": "John has no meetings the whole week.", "Jennifer": "Jennifer has meetings on Monday during 9:00 to 11:00, 11:30 to 13:00, 13:30 to 14:30, 15:00 to 17:00"}}}}
    Proposed Time: Monday, 13:00 - 13:30
    Verification: The proposed time works for both John and Jennifer.

    Example 2:
    Meeting Info: {{"participants": ["Patricia", "Harold"], "duration": "half an hour", "work_hours": "9:00 to 17:00", "days": ["Monday"], "schedules": {{"Patricia": "Patricia has blocked their calendar on Monday during 11:30 to 12:00, 12:30 to 13:00", "Harold": "Harold has meetings on Monday during 9:30 to 10:30, 11:30 to 12:00, 12:30 to 13:00, 13:30 to 15:30, 16:00 to 17:00"}}}}
    Proposed Time: Monday, 13:00 - 13:30
    Verification: The proposed time works for Patricia, but not for Harold.

    Meeting Info: {meeting_info}
    Proposed Time: {proposed_time}
    Verification:
    """
    return call_llm(prompt, system_instruction)

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response"""
    try:
        from google import genai
        from google.genai import types
        import os  # Import the 'os' module

        # Retrieve the API key from the environment variables
        gemini_api_key = os.environ.get("GEMINI_API_KEY")

        # Check if the API key is available
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        # Initialize the Gemini client
        client = genai.Client(api_key=gemini_api_key)

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