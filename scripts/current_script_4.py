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

def extract_meeting_data(question):
    """Extracts meeting data using LLM with structured extraction and validation."""
    system_instruction = "You are an expert at extracting meeting details."
    prompt = f"""
    Extract meeting details from the input. Output a JSON with 'participants', 'duration_minutes', 'days', and 'availability' (participant: day: [start-end]).

    Example:
    Input: Schedule John and Mary for 30 minutes on Monday between 9-5. John is busy 10-11, Mary is free.
    Output:
    {{
        "participants": ["John", "Mary"],
        "duration_minutes": 30,
        "days": ["Monday"],
        "availability": {{
            "John": {{ "Monday": ["9:00-10:00", "11:00-17:00"] }},
            "Mary": {{ "Monday": ["9:00-17:00"] }}
        }}
    }}
    
    Input: {question}
    """
    return call_llm(prompt, system_instruction)

def validate_extracted_data(extracted_data):
    """Validates the extracted data using LLM, correcting inconsistencies."""
    system_instruction = "You are an expert at validating meeting data."
    prompt = f"""
    Validate the extracted meeting data. Check for inconsistencies and correct them. Output a valid JSON.

    Example:
    Input:
    {{
        "participants": ["John", "Mary"],
        "duration_minutes": 30,
        "days": ["Mondays"],
        "availability": {{
            "John": {{ "Monday": ["9:00-10:00", "11:00-17:00"] }},
            "Mary": {{ "Monday": ["9:00-17:00"] }}
        }}
    }}
    Output:
    {{
        "participants": ["John", "Mary"],
        "duration_minutes": 30,
        "days": ["Monday"],
        "availability": {{
            "John": {{ "Monday": ["9:00-10:00", "11:00-17:00"] }},
            "Mary": {{ "Monday": ["9:00-17:00"] }}
        }}
    }}
    
    Input: {extracted_data}
    """
    return call_llm(prompt, system_instruction)

def find_meeting_time(validated_data):
    """Finds a valid meeting time based on validated data, using LLM for final suggestion."""
    system_instruction = "You are an expert meeting scheduler."
    prompt = f"""
    Find a valid meeting time based on the following validated meeting data. Suggest a meeting time or state if impossible.
    Prioritize the earliest available time.

    Example:
    Input:
    {{
        "participants": ["John", "Mary"],
        "duration_minutes": 30,
        "days": ["Monday"],
        "availability": {{
            "John": {{ "Monday": ["9:00-10:00", "11:00-17:00"] }},
            "Mary": {{ "Monday": ["9:00-17:00"] }}
        }}
    }}
    Output: Here is the proposed time: Monday, 9:00 - 9:30
    
    Input: {validated_data}
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule meetings."""
    try:
        # 1. Extract data
        extracted_data = extract_meeting_data(question)

        # 2. Validate and correct extracted data
        validated_data = validate_extracted_data(extracted_data)

        # 3. Find a meeting time
        meeting_time = find_meeting_time(validated_data)
        return meeting_time

    except Exception as e:
        return f"Error: {str(e)}"