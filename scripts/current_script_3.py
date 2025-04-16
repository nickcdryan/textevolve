import os
import json

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

def extract_meeting_constraints(question):
    """Extract all constraints using example-based prompting."""
    system_instruction = "You are an expert at identifying meeting constraints."
    prompt = f"""
    Identify and extract all meeting constraints (participants, duration, days, time range, preferences) from the given text.

    Example:
    Text: You need to schedule a meeting for John and Jane for half an hour between 9:00 to 17:00 on Monday. Jane prefers not to meet before 10:00.
    Constraints:
    {{
      "participants": ["John", "Jane"],
      "duration": "half an hour",
      "days": ["Monday"],
      "time_range": ["9:00", "17:00"],
      "preferences": {{"Jane": "not before 10:00"}}
    }}

    Now extract constraints from:
    {question}
    """
    try:
        response = call_llm(prompt, system_instruction)
        return json.loads(response)
    except Exception as e:
        print(f"Error extracting constraints: {e}")
        return None

def extract_participant_schedules(question, participants):
    """Extract schedules for each participant using LLM calls."""
    system_instruction = "You are an expert at extracting schedules for individuals from text."
    prompt = f"""
    Extract the schedule for each participant from the following text. Return a JSON object where each participant's name is a key, and the value is a list of their busy time slots.

    Example:
    Text: John is busy on Monday from 9:00 to 10:00 and on Tuesday from 14:00 to 15:00. Jane is busy on Wednesday from 11:00 to 12:00.
    Participants: ["John", "Jane"]
    Schedules:
    {{
      "John": [["Monday", "9:00", "10:00"], ["Tuesday", "14:00", "15:00"]],
      "Jane": [["Wednesday", "11:00", "12:00"]]
    }}

    Now, extract the schedules for the following participants: {participants} from this text:
    {question}
    """
    try:
        response = call_llm(prompt, system_instruction)
        return json.loads(response)
    except Exception as e:
        print(f"Error extracting schedules: {e}")
        return None

def verify_extracted_data(constraints, schedules, question):
    """Verify constraints and schedules using a LLM-based verification agent."""
    system_instruction = "You are a meticulous verifier, checking information extraction for accuracy and completeness."
    prompt = f"""
    Verify that the extracted constraints and schedules are complete, accurate, and consistent with the original text.
    If there are any errors or omissions, explain them in detail. If the extraction is perfect, respond with "VALID".

    Example:
    Text: Schedule a meeting for John and Jane on Tuesday. John is busy 9:00-10:00.
    Constraints:
    {{
      "participants": ["John", "Jane"],
      "duration": null,
      "days": ["Tuesday"],
      "time_range": ["9:00", "17:00"]
    }}
    Schedules:
    {{
      "John": [["Tuesday", "9:00", "10:00"]]
    }}
    Verification Result: VALID

    Now, verify the extracted constraints and schedules against the following text:
    Text: {question}
    Constraints: {json.dumps(constraints)}
    Schedules: {json.dumps(schedules)}
    """
    try:
        response = call_llm(prompt, system_instruction)
        return response
    except Exception as e:
        print(f"Error verifying data: {e}")
        return None

def find_available_time(constraints, schedules):
    """Find available time using LLM reasoning."""
    system_instruction = "You are an expert at finding available times given meeting constraints and schedules."
    prompt = f"""
    Given the meeting constraints and participant schedules, find a time that works for everyone.
    Respond in the format: "Here is the proposed time: [Day], [Start Time] - [End Time]". If no time is available, respond with "No available time found."

    Example:
    Constraints:
    {{
      "participants": ["John", "Jane"],
      "duration": "half an hour",
      "days": ["Monday"],
      "time_range": ["9:00", "17:00"]
    }}
    Schedules:
    {{
      "John": [["Monday", "9:00", "9:30"]],
      "Jane": [["Monday", "10:00", "10:30"]]
    }}
    Available Time: Here is the proposed time: Monday, 9:30 - 10:00

    Now, find an available time for the following:
    Constraints: {json.dumps(constraints)}
    Schedules: {json.dumps(schedules)}
    """
    try:
        response = call_llm(prompt, system_instruction)
        return response
    except Exception as e:
        print(f"Error finding available time: {e}")
        return None

def main(question):
    """Main function to schedule a meeting."""
    try:
        constraints = extract_meeting_constraints(question)
        if not constraints:
            return "Error: Could not extract meeting constraints."

        schedules = extract_participant_schedules(question, constraints["participants"])
        if not schedules:
            return "Error: Could not extract schedules."

        verification_result = verify_extracted_data(constraints, schedules, question)
        if "VALID" not in verification_result:
            return f"Error: Data verification failed: {verification_result}"

        available_time = find_available_time(constraints, schedules)
        if not available_time:
            return "Error: Could not find available time."

        return available_time

    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred while scheduling the meeting."