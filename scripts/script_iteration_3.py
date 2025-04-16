import os
import re
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

def extract_scheduling_info(text):
    """Extracts scheduling information from the input text using LLM with examples."""
    system_instruction = "You are an expert at extracting scheduling information."
    prompt = f"""
    Extract the key scheduling information from the text, including participants, duration, day, time range, and conflicts.

    Example Input:
    You need to schedule a meeting for Alice and Bob for 1 hour between 9:00 and 17:00 on Tuesday.
    Alice is busy on Tuesday from 10:00-12:00 and 14:00-15:00. Bob is busy from 11:00-13:00 and 16:00-17:00.

    Reasoning:
    1. Participants: Identify the participants (Alice, Bob).
    2. Duration: Find the meeting duration (1 hour).
    3. Day: Identify the day (Tuesday).
    4. Time Range: Find the start and end times (9:00, 17:00).
    5. Conflicts: List each participant's busy times.

    Extracted Information:
    {{
      "participants": ["Alice", "Bob"],
      "duration": "1 hour",
      "day": "Tuesday",
      "time_range_start": "9:00",
      "time_range_end": "17:00",
      "conflicts": {{
        "Alice": ["10:00-12:00", "14:00-15:00"],
        "Bob": ["11:00-13:00", "16:00-17:00"]
      }}
    }}

    Now extract the information from this text:
    {text}
    """
    return call_llm(prompt, system_instruction)

def verify_extracted_info(text, extracted_info):
    """Verifies the extracted information for accuracy and completeness using LLM with examples."""
    system_instruction = "You are a meticulous verifier ensuring information accuracy."
    prompt = f"""
    You are given the original text and extracted information. Verify the extracted information is accurate and complete.

    Example Input:
    Text: Schedule a meeting for Charlie and David for 30 minutes between 10:00 and 16:00 on Wednesday.
    Charlie is unavailable from 11:00-12:00. David is unavailable from 14:00-15:00.
    Extracted Info:
    {{
      "participants": ["Charlie", "David"],
      "duration": "30 minutes",
      "day": "Wednesday",
      "time_range_start": "10:00",
      "time_range_end": "16:00",
      "conflicts": {{
        "Charlie": ["11:00-12:00"],
        "David": ["14:00-15:00"]
      }}
    }}

    Reasoning:
    1. Participants: Are all participants listed correctly?
    2. Duration: Is the duration accurate?
    3. Day: Is the correct day identified?
    4. Time Range: Is the time range correct?
    5. Conflicts: Are all conflicts accurately listed for each participant?

    Verification Result:
    VALID: All extracted information is accurate and complete.

    Now verify the extracted information for this text:
    Text: {text}
    Extracted Info: {extracted_info}
    """
    verification_result = call_llm(prompt, system_instruction)
    return verification_result

def find_available_time_slot(extracted_info):
    """Finds an available time slot using LLM, considering extracted constraints."""
    system_instruction = "You are an expert in finding available meeting times."
    prompt = f"""
    Given the extracted scheduling information, find a suitable meeting time.

    Example Input:
    {{
      "participants": ["Eve", "Frank"],
      "duration": "45 minutes",
      "day": "Thursday",
      "time_range_start": "9:00",
      "time_range_end": "17:00",
      "conflicts": {{
        "Eve": ["10:00-11:00", "14:00-15:00"],
        "Frank": ["11:00-12:00", "16:00-17:00"]
      }}
    }}

    Reasoning:
    1. Parse the conflicts for each participant.
    2. Determine available time slots for each participant within the time range.
    3. Find an overlapping time slot that accommodates the meeting duration.

    Available Time:
    Thursday, 9:00-9:45
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to orchestrate the scheduling process."""
    try:
        # 1. Extract Scheduling Information
        extracted_info = extract_scheduling_info(question)
        
        # 2. Verify Extracted Information
        verification = verify_extracted_info(question, extracted_info)
        if "INVALID" in verification:
            return "Error: Extracted information is invalid."
        
        # 3. Find Available Time Slot
        available_time = find_available_time_slot(extracted_info)
        if "Available Time:" in available_time:
            proposed_time = available_time.split("Available Time:\n")[1].strip()
            return "Here is the proposed time: " + proposed_time
        else:
            return "Could not find a valid meeting time."

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return "Error occurred while scheduling."