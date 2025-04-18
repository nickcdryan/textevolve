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

def extract_meeting_info(question):
    """Extract key meeting information with structured examples."""
    system_instruction = "You are an expert at extracting meeting details."
    prompt = f"""
    Extract participant names, busy slots, and duration from the question. Structure the busy slots as a list of tuples (start_time, end_time) for each participant.

    Example:
    Question: Schedule for John, Jane. John busy 9:00-10:00, Jane busy 11:00-12:00, duration 30 minutes.
    Extraction:
    {{
      "participants": ["John", "Jane"],
      "busy_slots": {{
        "John": [("9:00", "10:00")],
        "Jane": [("11:00", "12:00")]
      }},
      "duration": "30 minutes"
    }}

    Question: {question}
    Extraction:
    """
    return call_llm(prompt, system_instruction)

def is_valid_meeting_time(meeting_info, proposed_time):
    """Verify the proposed meeting time against extracted information."""
    system_instruction = "You are a meeting time validator."
    prompt = f"""
    Given the meeting information and a proposed time, verify if the time is valid.
    Meeting Info:
    {meeting_info}
    Proposed Time: {proposed_time}

    Example:
    Meeting Info:
    {{
      "participants": ["John", "Jane"],
      "busy_slots": {{
        "John": [("9:00", "10:00")],
        "Jane": [("11:00", "12:00")]
      }},
      "duration": "30 minutes"
    }}
    Proposed Time: Monday, 10:30 - 11:00
    Reasoning: John is free, Jane is free. The time is valid.
    Result: VALID

    Meeting Info:
    {meeting_info}
    Proposed Time: {proposed_time}
    Reasoning:
    Result:
    """
    return call_llm(prompt, system_instruction)

def find_available_time(meeting_info):
    """Find an available meeting time using structured information."""
    system_instruction = "You are an expert at finding available times."
    prompt = f"""
    Find a suitable meeting time, considering participant availability and duration.
    Meeting Info:
    {meeting_info}

    Example:
    Meeting Info:
    {{
      "participants": ["John", "Jane"],
      "busy_slots": {{
        "John": [("9:00", "10:00")],
        "Jane": [("11:00", "12:00")]
      }},
      "duration": "30 minutes"
    }}
    Reasoning:
    John is free after 10:00, Jane is free before 11:00 and after 12:00.
    The earliest available time is Monday, 10:30 - 11:00.
    Result: Monday, 10:30 - 11:00

    Meeting Info:
    {meeting_info}
    Reasoning:
    Result:
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule meetings."""
    try:
        # 1. Extract meeting data
        meeting_info = extract_meeting_info(question)

        # 2. Find an available time
        available_time = find_available_time(meeting_info)

        # 3. Verify the solution
        is_valid = is_valid_meeting_time(meeting_info, available_time)

        if "VALID" in is_valid:
            return f"Here is the proposed time: {available_time}"
        else:
            return "No suitable time slots found."

    except Exception as e:
        return f"Error: {str(e)}"