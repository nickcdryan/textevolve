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

def extract_meeting_details(question):
    """Extracts meeting details using LLM with structured extraction and validation."""
    system_instruction = "You are an expert at extracting meeting details. Focus on high accuracy."
    prompt = f"""
    Extract meeting details from the input. Output a JSON with 'participants', 'duration_minutes', 'days', and 'availability' (participant: day: [start-end]).
    Also, note participant preferences as 'preferences' (participant: reason).

    Example:
    Input: Schedule John and Mary for 30 minutes on Monday between 9-5. John is busy 10-11, Mary is free. Mary prefers to meet before noon.
    Output:
    {{
        "participants": ["John", "Mary"],
        "duration_minutes": 30,
        "days": ["Monday"],
        "availability": {{
            "John": {{ "Monday": ["9:00-10:00", "11:00-17:00"] }},
            "Mary": {{ "Monday": ["9:00-17:00"] }}
        }},
        "preferences": {{
            "Mary": "to meet before noon"
        }}
    }}
    
    Input: {question}
    """
    return call_llm(prompt, system_instruction)

def find_meeting_time(meeting_details_json):
    """Finds the best meeting time using LLM, incorporating preferences and a verification step."""
    system_instruction = "You are an expert at scheduling meetings, focusing on earliest availability and constraint satisfaction."
    prompt = f"""
    Given these meeting details, find the *earliest* valid meeting time, respecting all availability and preferences.

    Example:
    Input:
    {{
        "participants": ["John", "Mary"],
        "duration_minutes": 30,
        "days": ["Monday"],
        "availability": {{
            "John": {{ "Monday": ["9:00-10:00", "11:00-17:00"] }},
            "Mary": {{ "Monday": ["9:00-17:00"] }}
        }},
        "preferences": {{
            "Mary": "to meet before noon"
        }}
    }}
    Reasoning: The earliest time is 9:00. John is available. Mary is available and prefers this time.
    Output: Here is the proposed time: Monday, 9:00 - 9:30

    Input: {meeting_details_json}
    """
    suggested_time = call_llm(prompt, system_instruction)
    return suggested_time

def verify_meeting_time(question, meeting_details_json, suggested_time):
    """Verifies if the suggested meeting time is valid and respects all constraints."""
    system_instruction = "You are a meticulous meeting scheduler. Double-check every detail."
    prompt = f"""
    Carefully verify if the suggested meeting time is valid and respects *all* availability constraints and preferences from the original question. If *any* constraint is violated, respond with "INVALID: [reason]". If the time is valid, respond with "VALID".

    Example:
    Question: Schedule John and Mary for 30 minutes on Monday between 9-5. John is busy 10-11, Mary is free. Mary prefers to meet before noon.
    Meeting Details:
    {{
        "participants": ["John", "Mary"],
        "duration_minutes": 30,
        "days": ["Monday"],
        "availability": {{
            "John": {{ "Monday": ["9:00-10:00", "11:00-17:00"] }},
            "Mary": {{ "Monday": ["9:00-17:00"] }}
        }},
        "preferences": {{
            "Mary": "to meet before noon"
        }}
    }}
    Suggested Time: Here is the proposed time: Monday, 11:00 - 11:30
    Reasoning: John is available. Mary is available and her preference is met. All constraints are respected.
    Output: VALID

    Question: {question}
    Meeting Details: {meeting_details_json}
    Suggested Time: {suggested_time}
    """
    verification_result = call_llm(prompt, system_instruction)
    return verification_result

def main(question):
    """Main function to schedule meetings with LLM and verification."""
    try:
        # 1. Extract meeting details.
        meeting_details_json = extract_meeting_details(question)

        # 2. Find a meeting time
        suggested_time = find_meeting_time(meeting_details_json)

        # 3. Verify the suggested time
        verification_result = verify_meeting_time(question, meeting_details_json, suggested_time)

        # 4. Return the result
        if "INVALID" in verification_result:
            return "No suitable meeting time found." #f"Error: {verification_result}"
        else:
            return suggested_time

    except Exception as e:
        return f"Error: {str(e)}"