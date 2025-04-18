import os
import json
import re
import datetime
from datetime import timedelta

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
    """Schedules meetings using a different approach: decompose problem into extraction, then use a verification and iterative refinement strategy to find available slots."""
    try:
        # 1. Extract meeting information
        extracted_info = extract_meeting_info(question)
        if "Error" in extracted_info:
            return "Error extracting meeting information."

        meeting_info = json.loads(extracted_info)

        # 2. Find an available meeting slot with validation
        available_slot = find_available_slot(meeting_info, question)
        if "Error" in available_slot:
            return "Error finding a suitable meeting time."

        return available_slot

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def extract_meeting_info(question):
    """Extracts meeting information with a few-shot example."""
    system_instruction = "You are an expert at extracting meeting scheduling details into a structured format."
    prompt = f"""
    Extract structured meeting information from the following text. Return the information as a JSON object.

    Example:
    Input: You need to schedule a meeting for John and Jane for 30 minutes between 9:00 and 17:00 on Monday. John is busy 10:00-11:00, Jane is busy 13:00-14:00.
    Output:
    {{
      "participants": ["John", "Jane"],
      "duration": 30,
      "days": ["Monday"],
      "work_hours": ["9:00", "17:00"],
      "schedules": {{
        "John": [["10:00", "11:00"]],
        "Jane": [["13:00", "14:00"]]
      }}
    }}

    Input: {question}
    Output:
    """
    try:
        extracted_info = call_llm(prompt, system_instruction)
        return extracted_info
    except Exception as e:
        return f"Error extracting info: {str(e)}"

def find_available_slot(meeting_info, question, max_attempts=5):
    """Finds available slots and validates them iteratively with LLM."""
    system_instruction = "You are an expert meeting scheduler. You will propose a time and then confirm with the validator that it works."

    for attempt in range(max_attempts):
        # 1. Propose a meeting slot
        proposal_prompt = f"""
        Based on this meeting information: {meeting_info}, propose a possible meeting slot (day, start time, end time). Consider work hours and participant schedules.

        Example:
        Meeting Info: {{"participants": ["John", "Jane"], "duration": 30, "days": ["Monday"], "work_hours": ["9:00", "17:00"], "schedules": {{"John": [["10:00", "11:00"]], "Jane": [["13:00", "14:00"]]}}}}
        Proposed Slot: Monday, 9:00 - 9:30

        Meeting Info: {meeting_info}
        Proposed Slot:
        """

        proposed_slot = call_llm(proposal_prompt, system_instruction)

        # 2. Validate proposed slot against constraints
        validation_prompt = f"""
        You are a meeting scheduler and need to determine if this slot: {proposed_slot} is valid given the following constraints: {meeting_info}.

        Respond with VALID or INVALID followed by the reason.

        Example:
        Proposed Slot: Monday, 9:00 - 9:30
        Meeting Info: {{"participants": ["John", "Jane"], "duration": 30, "days": ["Monday"], "work_hours": ["9:00", "17:00"], "schedules": {{"John": [["10:00", "11:00"]], "Jane": [["13:00", "14:00"]]}}}}
        Validation: VALID

        Proposed Slot: {proposed_slot}
        Meeting Info: {meeting_info}
        Validation:
        """

        validation_result = call_llm(validation_prompt, system_instruction)

        if "VALID" in validation_result:
            return f"Here is the proposed time: {proposed_slot}"
        else:
            continue  # Retry with a new proposal

    return "Error: Could not find a suitable meeting time after multiple attempts."