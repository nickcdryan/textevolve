import os
import re
import math

def main(question):
    """Schedules meetings using a structured approach with specialized agents and multi-stage verification."""
    try:
        # Step 1: Extract meeting information using the Extraction Agent with validation
        extracted_info = extract_meeting_info(question)
        if "Error" in extracted_info:
            return extracted_info

        # Step 2: Schedule the meeting using the Scheduling Agent with validation
        scheduled_meeting = schedule_meeting(extracted_info, question)
        if "Error" in scheduled_meeting:
            return scheduled_meeting

        return scheduled_meeting

    except Exception as e:
        return f"Error processing the request: {str(e)}"

def extract_meeting_info(question, max_attempts=3):
    """Extracts meeting details with multi-example prompting and LLM-based validation."""
    system_instruction = "You are an expert at extracting meeting details from text. Extract data, don't determine if the time works."

    for attempt in range(max_attempts):
        prompt = f"""
        You are an expert at extracting meeting details. Extract:
        - participants (list of names)
        - duration (integer, minutes)
        - days (list of strings, e.g., "Monday", "Tuesday")
        - existing schedules (dictionary, participant name -> list of time ranges "HH:MM-HH:MM")

        Example 1:
        Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9:00-10:00, Mary is busy from 11:00-12:00.
        Extraction: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": ["9:00-10:00"], "Mary": ["11:00-12:00"]}}}}

        Example 2:
        Question: Schedule a meeting for Alice, Bob, and Charlie for 1 hour on Tuesday and Wednesday. Alice is busy from 14:00-15:00 on Tuesday, Bob is busy from 10:00-11:00 on Wednesday. Charlie is free.
        Extraction: {{"participants": ["Alice", "Bob", "Charlie"], "duration": 60, "days": ["Tuesday", "Wednesday"], "schedules": {{"Alice": ["14:00-15:00 (Tuesday)"], "Bob": ["10:00-11:00 (Wednesday)"], "Charlie": []}}}}

        Example 3:
        Question: Schedule a meeting for Jonathan, Janice, Walter, Mary, Roger, Tyler and Arthur for half an hour on Monday. Jonathan is busy 9:30-10:00, 12:30-13:30, 14:30-15:00; Janice is busy 9:00-9:30, 11:30-12:00, 12:30-13:30, 14:30-15:00, 16:00-16:30; Walter is busy 9:30-10:00, 11:30-12:00; Mary is busy 12:00-12:30, 13:30-14:00; Roger is busy 9:30-10:30, 11:00-12:30, 13:00-13:30, 14:00-15:30, 16:00-16:30; Tyler is busy 9:30-11:00, 11:30-12:30, 13:30-14:00, 15:00-16:00; Arthur is busy 10:00-11:30, 12:30-13:00, 13:30-14:00, 14:30-16:00;
        Extraction: {{"participants": ["Jonathan", "Janice", "Walter", "Mary", "Roger", "Tyler", "Arthur"], "duration": 30, "days": ["Monday"], "schedules": {{"Jonathan": ["9:30-10:00", "12:30-13:30", "14:30-15:00"], "Janice": ["9:00-9:30", "11:30-12:00", "12:30-13:30", "14:30-15:00", "16:00-16:30"], "Walter": ["9:30-10:00", "11:30-12:00"], "Mary": ["12:00-12:30", "13:30-14:00"], "Roger": ["9:30-10:30", "11:00-12:30", "13:00-13:30", "14:00-15:30", "16:00-16:30"], "Tyler": ["9:30-11:00", "11:30-12:30", "13:30-14:00", "15:00-16:00"], "Arthur": ["10:00-11:30", "12:30-13:00", "13:30-14:00", "14:30-16:00"]}}}}

        Question: {question}
        Extraction:
        """
        extracted_info = call_llm(prompt, system_instruction)

        # LLM-based validation
        validation_prompt = f"""
        You are an expert at verifying extracted information. Given the question and extraction, verify:
        1. Are all participants identified?
        2. Is the duration correct?
        3. Are all days mentioned included?
        4. Are the schedules correctly associated with each participant and day?

        If EVERYTHING is correct, respond EXACTLY with "VALID".
        Otherwise, explain the errors.

        Question: {question}
        Extracted Info: {extracted_info}
        Verification:
        """
        validation_result = call_llm(validation_prompt, system_instruction)

        if "VALID" in validation_result:
            return extracted_info
        else:
            print(f"Extraction validation failed (attempt {attempt+1}): {validation_result}")
    return f"Error: Extraction failed after multiple attempts: {validation_result}"

def schedule_meeting(extracted_info, question):
    """Schedules a meeting given extracted information."""
    system_instruction = "You are an expert meeting scheduler. Given all the information, generate a final proposed time that works."
    prompt = f"""
    You are an expert at scheduling meetings. Given the question and extracted meeting details, return a final proposed time.
    Information:
    - Participants: list of names
    - Duration: integer, minutes
    - Days: list of strings, e.g., "Monday", "Tuesday"
    - Existing schedules: dictionary, participant name -> list of time ranges "HH:MM-HH:MM"

    Example 1:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy 9:00-10:00, Mary is busy 11:00-12:00.
    Extraction: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": ["9:00-10:00"], "Mary": ["11:00-12:00"]}}}}
    Reasoning: John is available 10:00-17:00. Mary is available 9:00-11:00 and 12:00-17:00. The best time is 10:00-10:30.
    Proposed Time: Here is the proposed time: Monday, 10:00-10:30

    Example 2:
    Question: Schedule a meeting for Alice, Bob, and Charlie for 1 hour on Tuesday and Wednesday. Alice is busy 14:00-15:00 on Tuesday, Bob is busy 10:00-11:00 on Wednesday. Charlie is free.
    Extraction: {{"participants": ["Alice", "Bob", "Charlie"], "duration": 60, "days": ["Tuesday", "Wednesday"], "schedules": {{"Alice": ["14:00-15:00 (Tuesday)"], "Bob": ["10:00-11:00 (Wednesday)"], "Charlie": []}}}}
    Reasoning: On Tuesday, Alice is busy 14:00-15:00. Bob and Charlie are free. A time that works is 10:00-11:00.
    Proposed Time: Here is the proposed time: Tuesday, 10:00-11:00

    Given this information:
    Extracted Info: {extracted_info}
    Question: {question}

    Respond ONLY in the format 'Here is the proposed time: [day], [start_time]-[end_time]'
    Proposed Time:
    """
    proposed_time = call_llm(prompt, system_instruction)
    return proposed_time

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