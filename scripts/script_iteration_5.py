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
    """Schedules meetings using a new approach: decompose problem into extraction, conflict validation and iterative propose and refine"""
    try:
        # 1. Extract meeting information
        extracted_info = extract_meeting_info(question)
        if "Error" in extracted_info:
            return "Error extracting meeting information."

        meeting_info = json.loads(extracted_info)

        # 2. Iteratively propose and refine a meeting slot
        meeting_slot = propose_and_refine_slot(meeting_info, question)
        if "Error" in meeting_slot:
            return "Error finding suitable meeting time."

        return meeting_slot

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def extract_meeting_info(question):
    """Extract meeting information using LLM with an embedded example."""
    system_instruction = "You are an expert at extracting information from meeting scheduling requests."
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

def propose_and_refine_slot(meeting_info, question, max_attempts=5):
    """Iteratively proposes and refines a meeting slot using LLM until a valid slot is found."""
    system_instruction = "You are an expert meeting scheduler, iteratively refining proposed meeting times based on constraints."

    for attempt in range(max_attempts):
        # 1. Propose a meeting slot
        proposal_prompt = f"""
        Based on this meeting information: {meeting_info}, propose a possible meeting slot (day, start time, end time).
        Be mindful of work hours and known participant schedules.

        Example:
        Meeting Info: {{"participants": ["John", "Jane"], "duration": 30, "days": ["Monday"], "work_hours": ["9:00", "17:00"], "schedules": {{"John": [["10:00", "11:00"]], "Jane": [["13:00", "14:00"]]}}}}
        Proposed Slot: Monday, 9:00 - 9:30

        Meeting Info: {meeting_info}
        Proposed Slot:
        """

        proposed_slot = call_llm(proposal_prompt, system_instruction)

        # 2. Validate proposed slot against constraints
        validation_prompt = f"""
        You are an expert meeting scheduler. Validate that this meeting slot: {proposed_slot}
        works for everyone and satisfies these meeting requirements: {meeting_info}. Original problem: {question}

        Example:
        Proposed Slot: Monday, 9:00 - 9:30
        Meeting Info: {{"participants": ["John", "Jane"], "duration": 30, "days": ["Monday"], "work_hours": ["9:00", "17:00"], "schedules": {{"John": [["10:00", "11:00"]], "Jane": [["13:00", "14:00"]]}}}}
        Validation: Valid

        Proposed Slot: {proposed_slot}
        Meeting Info: {meeting_info}
        Validation:
        """

        validation_result = call_llm(validation_prompt, system_instruction)

        if "Valid" in validation_result:
            return f"Here is the proposed time: {proposed_slot}"
        else:
            # 3. Refine slot if invalid
            refinement_prompt = f"""
            You are an expert meeting scheduler. The proposed meeting slot {proposed_slot} is invalid because: {validation_result}.
            Suggest a different meeting slot that resolves the issues.
            Meeting Info: {meeting_info}
            Original problem: {question}

            Example:
            Proposed Slot: Monday, 9:00 - 9:30
            Meeting Info: {{"participants": ["John", "Jane"], "duration": 30, "days": ["Monday"], "work_hours": ["9:00", "17:00"], "schedules": {{"John": [["10:00", "11:00"]], "Jane": [["13:00", "14:00"]]}}}}
            Refined Slot: Monday, 11:00 - 11:30

            Proposed Slot: {proposed_slot}
            Meeting Info: {meeting_info}
            Refined Slot:
            """
            continue  # Retry with a new proposal

    return "Error: Could not find a suitable meeting time after multiple attempts."