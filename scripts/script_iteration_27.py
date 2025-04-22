import os
import re
import math

def main(question):
    """Schedules meetings using a new approach: structured decomposition into availability intervals, followed by LLM-driven reasoning and slot selection.

    This approach explicitly represents participant availability as time intervals and reasons over these.

    HYPOTHESIS: Representing availability as explicit intervals will improve constraint handling accuracy.
    """
    try:
        # 1. Extract structured meeting information, including participant availability intervals
        meeting_info = extract_meeting_info(question)
        if "Error" in meeting_info:
            return meeting_info

        # 2. Propose a meeting time using the extracted information
        proposed_time = propose_meeting_time(meeting_info, question)
        if "Error" in proposed_time:
            return proposed_time

        return proposed_time

    except Exception as e:
        return f"Error processing the request: {str(e)}"

def extract_meeting_info(question, max_attempts=3):
    """Extracts meeting details and represents participant availability as structured intervals."""
    system_instruction = "You are an expert at extracting structured meeting information."

    prompt = f"""
    You are an expert at extracting meeting details. Extract the following:
    - participants: list of names
    - duration: meeting duration in minutes
    - days: list of days
    - availabilities: each participant's availability as a list of TIME INTERVALS on specified days.

    Example 1:
    Question: Schedule John and Mary for 30 minutes on Monday. John is busy 9:00-10:00, Mary 11:00-12:00.
    Extraction:
    {{
        "participants": ["John", "Mary"],
        "duration": 30,
        "days": ["Monday"],
        "availabilities": {{
            "John": [["10:00", "17:00"]],  // Assuming work hours 9-17
            "Mary": [["9:00", "11:00"], ["12:00", "17:00"]]
        }}
    }}

    Example 2:
    Question: Schedule Alice, Bob for 1 hour on Tuesday/Wednesday. Alice is busy 14:00-15:00 Tue, Bob is free.
    Extraction:
    {{
        "participants": ["Alice", "Bob"],
        "duration": 60,
        "days": ["Tuesday", "Wednesday"],
        "availabilities": {{
            "Alice": [["9:00", "14:00", "Tuesday"], ["15:00", "17:00", "Tuesday"], ["9:00", "17:00", "Wednesday"]],
            "Bob": [["9:00", "17:00", "Tuesday"], ["9:00", "17:00", "Wednesday"]]
        }}
    }}

    Question: {question}
    Extraction:
    """
    extracted_info = call_llm(prompt, system_instruction)

    # Basic validation (can enhance with more checks)
    if "Error" in extracted_info or not isinstance(extracted_info, str):
        return f"Error: Extraction failed. Result: {extracted_info}"
    return extracted_info

def propose_meeting_time(meeting_info, question):
    """Proposes a meeting time given extracted structured data."""
    system_instruction = "You are an expert at proposing meeting times given availability data."
    prompt = f"""
    You are an expert at scheduling meetings. Given the extracted meeting details, determine and return a valid proposed meeting time.

    Example:
    Meeting Info:
    {{
        "participants": ["John", "Mary"],
        "duration": 30,
        "days": ["Monday"],
        "availabilities": {{
            "John": [["10:00", "17:00"]],
            "Mary": [["9:00", "11:00"], ["12:00", "17:00"]]
        }}
    }}
    Question: Schedule John and Mary for 30 minutes on Monday. John is busy 9:00-10:00, Mary 11:00-12:00.
    Reasoning: Both are available after 10:00. A 30 min slot works at 10:00.
    Proposed Time: Here is the proposed time: Monday, 10:00-10:30

    Meeting Info: {meeting_info}
    Question: {question}
    Reasoning:
    Proposed Time:
    """
    proposed_time = call_llm(prompt, system_instruction)

    # Basic format validation
    if "Here is the proposed time:" not in proposed_time:
        return f"Error: Invalid format in proposed time. Result: {proposed_time}"
    return proposed_time

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