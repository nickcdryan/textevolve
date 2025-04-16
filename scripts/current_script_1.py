import os
import re
import json

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
    """Extract meeting details (participants, duration, days) from the question using LLM with example."""
    system_instruction = "You are an expert at extracting meeting details."
    prompt = f"""
    Extract the participants, duration, and possible days for the meeting from the given text.

    Example:
    Text: You need to schedule a meeting for Carol and Mark for half an hour between the work hours of 9:00 to 17:00 on Monday.
    Extracted Info:
    {{
        "participants": ["Carol", "Mark"],
        "duration": "30 minutes",
        "days": ["Monday"]
    }}

    Text: You need to schedule a meeting for Jennifer and Christine for half an hour between the work hours of 9:00 to 17:00 on either Monday, Tuesday or Wednesday.
    Extracted Info:
    {{
        "participants": ["Jennifer", "Christine"],
        "duration": "30 minutes",
        "days": ["Monday", "Tuesday", "Wednesday"]
    }}

    Now extract from this text:
    {question}
    """
    try:
        response = call_llm(prompt, system_instruction)
        return json.loads(response)
    except Exception as e:
        print(f"Error extracting meeting info: {e}")
        return None

def extract_schedules(question, participants, days):
    """Extract and verify schedules for each participant on the specified days using LLM with example."""
    system_instruction = "You are an expert at extracting participant schedules."
    prompt = f"""
    Extract the schedules for each participant on the specified days. Verify the extracted schedules for correctness.

    Example:
    Question: You need to schedule a meeting for Carol and Mark for half an hour on Monday.
    Here are the existing schedules: Carol has blocked their calendar on Monday during 10:00 to 11:00; Mark has blocked their calendar on Monday during 9:30 to 10:00.
    Participants: ["Carol", "Mark"]
    Days: ["Monday"]
    Extracted Schedules:
    {{
        "Carol": {{
            "Monday": ["10:00-11:00"]
        }},
        "Mark": {{
            "Monday": ["9:30-10:00"]
        }}
    }}

    Now extract from this question, participants and days:
    Question: {question}
    Participants: {participants}
    Days: {days}
    """
    try:
        response = call_llm(prompt, system_instruction)
        return json.loads(response)
    except Exception as e:
        print(f"Error extracting schedules: {e}")
        return None

def find_available_time(meeting_info, schedules):
    """Find an available time slot that works for all participants using LLM with example."""
    system_instruction = "You are an expert at finding available meeting times."
    prompt = f"""
    Given the meeting information and participant schedules, find an available time slot that works for everyone.

    Example:
    Meeting Info:
    {{
        "participants": ["Carol", "Mark"],
        "duration": "30 minutes",
        "days": ["Monday"]
    }}
    Schedules:
    {{
        "Carol": {{
            "Monday": ["10:00-11:00"]
        }},
        "Mark": {{
            "Monday": ["9:30-10:00"]
        }}
    }}
    Available Time: Monday, 9:00 - 9:30

    Now find an available time for the following:
    Meeting Info: {meeting_info}
    Schedules: {schedules}
    """
    try:
        response = call_llm(prompt, system_instruction)
        return response
    except Exception as e:
        print(f"Error finding available time: {e}")
        return None

def main(question):
    """Main function to schedule a meeting."""
    try:
        meeting_info = extract_meeting_info(question)
        if not meeting_info:
            return "Could not extract meeting information."

        schedules = extract_schedules(question, meeting_info["participants"], meeting_info["days"])
        if not schedules:
            return "Could not extract schedules."

        available_time = find_available_time(meeting_info, schedules)
        if not available_time:
            return "Could not find an available time."

        return f"Here is the proposed time: {available_time}"

    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred while scheduling the meeting."