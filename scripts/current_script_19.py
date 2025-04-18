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

def extract_meeting_data(question):
    """Extract participant names, constraints, and duration from the question using LLM with examples."""
    system_instruction = "You are an expert at extracting meeting details."
    prompt = f"""
    Extract the participant names, constraints, and duration from the question.

    Example:
    Question: Schedule a meeting for John, Jane, and Mike for 30 minutes. John is busy Monday 9-10, Jane prefers Tuesdays.
    Extracted Data:
    {{
        "participants": ["John", "Jane", "Mike"],
        "constraints": "John is busy Monday 9-10, Jane prefers Tuesdays",
        "duration": 30
    }}

    Question: {question}
    Extracted Data:
    """
    return call_llm(prompt, system_instruction)

def find_available_time(participants, constraints, duration):
    """Find a suitable meeting time using LLM, incorporating constraints and duration, with examples."""
    system_instruction = "You are an expert at finding available meeting times."
    prompt = f"""
    Given the participants, constraints, and duration, find a suitable meeting time. Prioritize earlier times.

    Example:
    Participants: ["John", "Jane"]
    Constraints: John is busy Monday 9-10, Jane prefers Tuesdays.
    Duration: 30 minutes
    Solution: Tuesday, 11:00 - 11:30

    Participants: {participants}
    Constraints: {constraints}
    Duration: {duration} minutes
    Solution:
    """
    return call_llm(prompt, system_instruction)

def is_valid_meeting_time(question, proposed_time):
    """Verify if the proposed meeting time is valid given the original question, with examples."""
    system_instruction = "You are an expert at verifying meeting times."
    prompt = f"""
    Verify if the proposed meeting time is valid given the original question. Consider all constraints.

    Example:
    Question: Schedule a meeting for John, Jane, and Mike. John is busy Monday 9-10.
    Proposed Time: Monday, 11:00 - 11:30
    Verification: VALID

    Question: {question}
    Proposed Time: {proposed_time}
    Verification:
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule meetings, refactored to use the new function design."""
    try:
        meeting_data = extract_meeting_data(question)
        if "Error" in meeting_data:
            return meeting_data  # Propagate the error

        try:
            meeting_data_json = json.loads(meeting_data)
            participants = meeting_data_json.get("participants", [])
            constraints = meeting_data_json.get("constraints", "")
            duration = meeting_data_json.get("duration", 30)  # Default to 30 minutes

        except json.JSONDecodeError as e:
            return f"Error decoding JSON: {str(e)}"

        available_time = find_available_time(participants, constraints, duration)
        if "Error" in available_time:
            return available_time  # Propagate the error

        verification = is_valid_meeting_time(question, available_time)
        if "Error" in verification:
            return verification  # Propagate the error
        
        if "VALID" in verification:
            return f"Here is the proposed time: {available_time}"
        else:
            return "No suitable time slots found."

    except Exception as e:
        return f"Error: {str(e)}"