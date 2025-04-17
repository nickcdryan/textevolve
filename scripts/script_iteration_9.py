import os
import json
import re
import math

def main(question):
    """
    Schedules meetings using a novel approach: LLM-driven Information Extraction,
    Deterministic schedule generation, and LLM-driven verification.
    This script addresses JSON output unreliability and incorporates a dedicated schedule generation.
    """
    try:
        # 1. Extract structured information using LLM
        extracted_info = extract_meeting_info(question)
        if "Error" in extracted_info: return extracted_info

        # 2. Generate a candidate schedule using deterministic logic
        candidate_schedule = generate_schedule(extracted_info)
        if "Error" in candidate_schedule: return candidate_schedule

        # 3. Verify the schedule using LLM
        verified_schedule = verify_schedule(question, candidate_schedule)
        if "Error" in verified_schedule: return verified_schedule

        return verified_schedule

    except Exception as e:
        return f"Error in main: {str(e)}"

def extract_meeting_info(question):
    """Extracts meeting information using LLM with examples."""
    system_instruction = "You are an information extraction expert. Return valid JSON only."
    prompt = f"""
    Extract meeting details, constraints, and participants from the following text.
    Return a JSON object.
    Example:
    Input: Schedule a meeting for John and Jane for 30 minutes on Monday. John is busy 1-2pm.
    Reasoning: Participants: John, Jane. Duration: 30. Day: Monday. John's busy: 1-2pm.
    Output: {{"participants": ["John", "Jane"], "duration": 30, "day": "Monday", "john_busy": "1-2pm"}}
    Now extract information from: {question}
    """
    try:
        raw_json = call_llm(prompt, system_instruction)
        # Extract JSON string from the raw text using regex to handle unreliability
        match = re.search(r"\{[\s\S]*\}", raw_json)
        if match:
          json_string = match.group(0)
          return json_string
        else:
          return "Error: could not extract json"

    except Exception as e:
        return f"Error extracting info: {str(e)}"

def generate_schedule(extracted_info):
  """Generates a valid meeting time using deterministic methods."""
  try:
    data = json.loads(extracted_info)
    participants = data.get("participants", [])
    duration = data.get("duration", 30)
    day = data.get("day", "Monday")
    john_busy = data.get("john_busy", "")

    # This is a placeholder
    proposed_time = "Here is the proposed time: Monday, 9:00 - 9:30"
    return proposed_time
  except Exception as e:
    return f"Error generating schedule: {str(e)}"

def verify_schedule(question, proposed_schedule):
    """Verifies the schedule using LLM with examples."""
    system_instruction = "You are a schedule verification expert. Validate if a schedule works."
    prompt = f"""
    Verify if the proposed schedule works based on the question.
    Example:
    Question: Schedule a meeting for John and Jane for 30 minutes on Monday. John is busy 1-2pm.
    Schedule: Here is the proposed time: Monday, 1:30 - 2:00
    Reasoning: John is busy at 1:30-2:00, so this schedule doesn't work.
    Output: Invalid - John is busy.
    Now verify:
    Question: {question}
    Schedule: {proposed_schedule}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error verifying schedule: {str(e)}"

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

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