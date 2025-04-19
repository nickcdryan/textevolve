import os
import re
import json
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

def extract_meeting_constraints(text):
    """Extract meeting constraints using LLM with examples."""
    system_instruction = "You are an expert at extracting meeting constraints from text."
    prompt = f"""
    Extract all the constraints from the following text.

    Example:
    Text: You need to schedule a meeting for Joyce, Christine and Alexander for half an hour between the work hours of 9:00 to 17:00 on Monday. Joyce has meetings on Monday during 11:00 to 11:30, 13:30 to 14:00, 14:30 to 16:30; Christinehas no meetings the whole day. Alexander has meetings on Monday during 9:00 to 11:00, 12:00 to 12:30, 13:30 to 15:00, 15:30 to 16:00, 16:30 to 17:00; Christine can not meet on Monday before 12:00.
    Extracted Constraints:
    {{
        "participants": ["Joyce", "Christine", "Alexander"],
        "duration": "half an hour",
        "day": "Monday",
        "start_time": "9:00",
        "end_time": "17:00",
        "Joyce_schedule": ["11:00 to 11:30", "13:30 to 14:00", "14:30 to 16:30"],
        "Christine_schedule": [],
        "Alexander_schedule": ["9:00 to 11:00", "12:00 to 12:30", "13:30 to 15:00", "15:30 to 16:00", "16:30 to 17:00"],
        "Christine_constraint": "not before 12:00"
    }}

    Text: {text}
    Extracted Constraints:
    """
    return call_llm(prompt, system_instruction)

def find_available_time(constraints_json):
    """Find an available time slot given the extracted constraints using LLM."""
    system_instruction = "You are an expert at scheduling meetings given a set of constraints."
    prompt = f"""
    Given the following constraints, find an available time slot that works for everyone.

    Example:
    Constraints:
    {{
        "participants": ["Joyce", "Christine", "Alexander"],
        "duration": "half an hour",
        "day": "Monday",
        "start_time": "9:00",
        "end_time": "17:00",
        "Joyce_schedule": ["11:00 to 11:30", "13:30 to 14:00", "14:30 to 16:30"],
        "Christine_schedule": [],
        "Alexander_schedule": ["9:00 to 11:00", "12:00 to 12:30", "13:30 to 15:00", "15:30 to 16:00", "16:30 to 17:00"],
        "Christine_constraint": "not before 12:00"
    }}
    Available Time: Monday, 12:30 - 13:00

    Constraints:
    {constraints_json}
    Available Time:
    """
    return call_llm(prompt, system_instruction)

def verify_solution(question, constraints_json, proposed_solution):
    """Verify if the proposed solution satisfies the constraints using LLM."""
    system_instruction = "You are a meeting scheduling expert. Verify proposed solutions meet constraints."
    prompt = f"""
    You are given a question and a proposed solution, along with extracted constraints from the question. Determine if the proposed solution is valid based on the constraints.

    Example:
    Question: You need to schedule a meeting for Joyce, Christine and Alexander for half an hour between the work hours of 9:00 to 17:00 on Monday. Joyce has meetings on Monday during 11:00 to 11:30, 13:30 to 14:00, 14:30 to 16:30; Christine has no meetings the whole day. Alexander has meetings on Monday during 9:00 to 11:00, 12:00 to 12:30, 13:30 to 15:00, 15:30 to 16:00, 16:30 to 17:00; Christine can not meet on Monday before 12:00.
    Constraints:
    {{
        "participants": ["Joyce", "Christine", "Alexander"],
        "duration": "half an hour",
        "day": "Monday",
        "start_time": "9:00",
        "end_time": "17:00",
        "Joyce_schedule": ["11:00 to 11:30", "13:30 to 14:00", "14:30 to 16:30"],
        "Christine_schedule": [],
        "Alexander_schedule": ["9:00 to 11:00", "12:00 to 12:30", "13:30 to 15:00", "15:30 to 16:00", "16:30 to 17:00"],
        "Christine_constraint": "not before 12:00"
    }}
    Proposed Solution: Monday, 12:30 - 13:00
    Is the solution valid? Yes, the solution is valid.

    Question: {question}
    Constraints:
    {constraints_json}
    Proposed Solution: {proposed_solution}
    Is the solution valid?
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule meetings."""
    try:
        # Extract constraints
        constraints_json = extract_meeting_constraints(question)

        # Find available time
        proposed_solution = find_available_time(constraints_json)

        # Verify solution
        verification_result = verify_solution(question, constraints_json, proposed_solution)

        if "Yes" in verification_result:
            return "Here is the proposed time: " + proposed_solution
        else:
            return "Could not find a valid meeting time."

    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error processing the request."