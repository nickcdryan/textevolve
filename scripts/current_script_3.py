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

def main(question):
    """Main function to schedule meetings using multi-agent approach."""
    try:
        # 1. Delegate to an extraction agent for structured data.
        extracted_data = extract_data(question)

        # 2. Delegate to a constraint checker agent for verification.
        constraints_valid = check_constraints(extracted_data)

        # 3. Delegate to a scheduling agent for potential meeting times.
        suggested_time = suggest_meeting_time(extracted_data)

        return suggested_time

    except Exception as e:
        return f"Error: {str(e)}"

def extract_data(question):
    """Extracts relevant data from the question using LLM."""
    system_instruction = "You are an expert data extractor for meeting scheduling."
    prompt = f"""
    Extract structured data required for meeting scheduling.

    Example:
    Input: Schedule a meeting for John and Mary for half an hour between 9:00 and 17:00 on Monday. John is busy 10:00-11:00, Mary is free.
    Reasoning: I need to extract participants, duration, working hours, possible days, and schedules.
    Output:
    {{
        "participants": ["John", "Mary"],
        "duration": "30 minutes",
        "working_hours": ["9:00", "17:00"],
        "possible_days": ["Monday"],
        "schedules": {{
            "John": {{"Monday": ["10:00-11:00"]}},
            "Mary": {{"Monday": []}}
        }}
    }}

    Input: {question}
    """
    return call_llm(prompt, system_instruction)

def check_constraints(extracted_data):
    """Checks and verifies extracted constraints using LLM."""
    system_instruction = "You are a constraint verification expert."
    prompt = f"""
    Verify that the extracted constraints are valid and consistent.

    Example:
    Input:
    {{
        "participants": ["John", "Mary"],
        "duration": "30 minutes",
        "working_hours": ["9:00", "17:00"],
        "possible_days": ["Monday"],
        "schedules": {{
            "John": {{"Monday": ["10:00-11:00"]}},
            "Mary": {{"Monday": []}}
        }}
    }}
    Reasoning: I need to make sure that the working hours and schedules match up. All participants and days are valid.
    Output: Constraints are valid.

    Input: {extracted_data}
    """
    return call_llm(prompt, system_instruction)

def suggest_meeting_time(extracted_data):
    """Suggests an appropriate meeting time using LLM based on extracted data."""
    system_instruction = "You are a meeting scheduling expert, skilled at finding suitable times."
    prompt = f"""
    Suggest a suitable meeting time based on these constraints.
    
    Example:
    Input:
    {{
        "participants": ["John", "Mary"],
        "duration": "30 minutes",
        "working_hours": ["9:00", "17:00"],
        "possible_days": ["Monday"],
        "schedules": {{
            "John": {{"Monday": ["10:00-11:00"]}},
            "Mary": {{"Monday": []}}
        }}
    }}
    Reasoning: Mary is free all day. John is busy from 10:00-11:00. Thus, any other time would work.
    Output: Here is the proposed time: Monday, 9:00 - 9:30

    Input: {extracted_data}
    """
    return call_llm(prompt, system_instruction)