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

def extract_meeting_info(question, max_attempts=3):
    """Extract meeting information from the question using LLM with examples and validation loop."""
    system_instruction = "You are an expert at extracting meeting details with structured output."

    for attempt in range(max_attempts):
        prompt = f"""
        Extract the following information from the text: participants, duration, constraints, schedules, preferences. Return a plain-text key:value format (no JSON).

        Example 1:
        Text: Schedule a meeting for John and Jennifer for half an hour between 9:00 and 17:00 on Monday. John has no meetings. Jennifer has meetings on Monday from 9:00-11:00.
        Extraction:
        participants: John, Jennifer
        duration: 30 minutes
        constraints: 9:00 to 17:00, Monday
        schedules: John - None, Jennifer - Monday 9:00-11:00
        preferences: None

        Example 2:
        Text: Schedule a meeting for Patricia and Harold for half an hour on Monday. Patricia is busy 11:30-12:00. Harold is busy 9:30-10:30 and 11:30-12:00. Harold prefers not to meet after 14:00.
        Extraction:
        participants: Patricia, Harold
        duration: 30 minutes
        constraints: Monday
        schedules: Patricia - Monday 11:30-12:00, Harold - Monday 9:30-10:30, 11:30-12:00
        preferences: Harold - not after 14:00

        Text: {question}
        Extraction:
        """

        extraction_result = call_llm(prompt, system_instruction)

        # Verification step: check if all required fields are present
        verification_prompt = f"""
        Verify that the following extraction contains information on participants, duration, constraints, and schedules.
        If any information is missing, identify what is missing.

        Extraction:
        {extraction_result}
        """

        verification_result = call_llm(verification_prompt)

        if "missing" not in verification_result.lower(): # Simple check. Replace with more sophisticated check as needed
            return extraction_result
        else:
            print(f"Missing information in extraction, attempt {attempt+1}/{max_attempts}: {verification_result}")

    return "Error: Could not reliably extract meeting information."

def schedule_meeting(meeting_info):
    """Schedule the meeting using LLM based on the extracted information with examples."""
    system_instruction = "You are an expert at scheduling meetings based on provided information."

    prompt = f"""
    Given the meeting information, schedule the meeting.

    Example:
    Meeting Information:
    participants: John, Jennifer
    duration: 30 minutes
    constraints: 9:00 to 17:00, Monday
    schedules: John - None, Jennifer - Monday 9:00-11:00
    preferences: None
    Schedule: Monday, 13:00 - 13:30

    Meeting Information: {meeting_info}
    Schedule:
    """

    schedule_result = call_llm(prompt, system_instruction)
    return schedule_result

def main(question):
    """Main function to orchestrate meeting scheduling."""
    meeting_info = extract_meeting_info(question)

    if "Error:" in meeting_info:
        return meeting_info  # Propagate error from extraction

    schedule = schedule_meeting(meeting_info)
    return schedule