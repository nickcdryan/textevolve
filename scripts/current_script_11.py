import os
import json
import re

def main(question):
    """Schedules a meeting by extracting information with LLMs and finding a valid time."""
    try:
        meeting_details = extract_meeting_details(question)
        if not meeting_details:
            return "Could not extract meeting details."

        available_time = find_available_time(meeting_details)
        if not available_time:
            return "No available time slots found."

        return f"Here is the proposed time: {available_time['day']}, {available_time['start']} - {available_time['end']}"

    except Exception as e:
        return f"An error occurred: {str(e)}"

def extract_meeting_details(question):
    """Extracts meeting details using LLM with embedded examples."""
    system_instruction = "You are an expert at extracting meeting details from text."
    prompt = f"""
    Extract the participants, duration, available days, and schedules from the text.

    Example:
    Text: You need to schedule a meeting for Ann and Sharon for one hour between 9:00 to 17:00 on Monday, Tuesday, or Wednesday. Ann is busy on Monday 11:30-12:00. Sharon is busy on Tuesday 9:30-10:00.
    Reasoning:
    1. Identify participants: Ann and Sharon.
    2. Determine duration: one hour.
    3. Extract available days: Monday, Tuesday, Wednesday.
    4. Parse Ann's schedule: Monday 11:30-12:00.
    5. Parse Sharon's schedule: Tuesday 9:30-10:00.
    Output:
    {{
        "participants": ["Ann", "Sharon"],
        "duration": "1 hour",
        "available_days": ["Monday", "Tuesday", "Wednesday"],
        "schedules": {{
            "Ann": {{"Monday": ["11:30-12:00"]}},
            "Sharon": {{"Tuesday": ["9:30-10:00"]}}
        }}
    }}

    Text: {question}
    """
    try:
        llm_response = call_llm(prompt, system_instruction)
        return json.loads(llm_response)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}, LLM Response: {llm_response}")
        return None
    except Exception as e:
        print(f"Error extracting details: {e}")
        return None

def find_available_time(meeting_details):
    """Finds an available time slot using LLM with embedded examples."""
    system_instruction = "You are an expert meeting scheduler."
    prompt = f"""
    Given meeting details, find an available time slot.

    Example:
    Details:
    {{
        "participants": ["Ann", "Sharon"],
        "duration": "1 hour",
        "available_days": ["Monday", "Tuesday"],
        "schedules": {{
            "Ann": {{"Monday": ["11:00-12:00"]}},
            "Sharon": {{"Tuesday": ["14:00-15:00"]}}
        }}
    }}
    Reasoning:
    1. Consider Ann's schedule on Monday. The meeting cannot be scheduled 11:00-12:00.
    2. Consider Sharon's schedule on Tuesday. The meeting cannot be scheduled 14:00-15:00.
    3. Find a time that works for all participants, let's choose Monday 9:00 - 10:00.
    Output:
    {{
        "day": "Monday",
        "start": "9:00",
        "end": "10:00"
    }}

    Details: {json.dumps(meeting_details)}
    """
    try:
        llm_response = call_llm(prompt, system_instruction)
        return json.loads(llm_response)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}, LLM Response: {llm_response}")
        return None
    except Exception as e:
        print(f"Error finding time: {e}")
        return None
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