import os
import re
import math

def main(question):
    """Schedules meetings using extraction and scheduling agents with enhanced validation and error handling."""
    try:
        extracted_info = extract_meeting_info(question)
        if "Error" in extracted_info:
            return extracted_info

        scheduled_meeting = schedule_meeting(extracted_info, question)
        if "Error" in scheduled_meeting:
            return scheduled_meeting

        return scheduled_meeting

    except Exception as e:
        return f"Error processing the request: {str(e)}"

def extract_meeting_info(question, max_attempts=3):
    """Extracts meeting details (participants, duration, days, schedules) using a specialized extraction agent with multi-example prompting and verification."""
    system_instruction = "You are an expert at extracting meeting details from text. Focus on extracting the data. Do not make assumptions or apply any scheduling logic."

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
        Question: You need to schedule a meeting for Jonathan, Janice, Walter, Mary, Roger, Tyler and Arthur for half an hour between the work hours of 9:00 to 17:00 on Monday. Jonathan has meetings on Monday during 9:30 to 10:00, 12:30 to 13:30, 14:30 to 15:00; Janice has blocked their calendar on Monday during 9:00 to 9:30, 11:30 to 12:00, 12:30 to 13:30, 14:30 to 15:00, 16:00 to 16:30.
        Extraction: {{"participants": ["Jonathan", "Janice"], "duration": 30, "days": ["Monday"], "schedules": {{"Jonathan": ["9:30-10:00", "12:30-13:30", "14:30-15:00"], "Janice": ["9:00-9:30", "11:30-12:00", "12:30-13:30", "14:30-15:00", "16:00-16:30"]}}}}

        Question: {question}
        Extraction:
        """
        extracted_info = call_llm(prompt, system_instruction)

        validation_prompt = f"""
        You are an expert at verifying extracted information. Verify:
        1. Are all participants identified?
        2. Is the duration correct?
        3. Are all days included?
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
    system_instruction = "You are an expert meeting scheduler. Use the extracted information to propose a final time that works."
    prompt = f"""
    You are an expert at scheduling meetings. Given the extracted meeting details, return a final proposed time.
    - Participants: list of names
    - Duration: integer, minutes
    - Days: list of strings, e.g., "Monday", "Tuesday"
    - Existing schedules: dictionary, participant name -> list of time ranges "HH:MM-HH:MM"

    Example 1:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9:00-10:00, Mary is busy from 11:00-12:00.
    Extracted Info: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": ["9:00-10:00"], "Mary": ["11:00-12:00"]}}}}
    Reasoning: John is available after 10:00. Mary is available before 11:00 and after 12:00. 10:00-10:30 works for both.
    Proposed Time: Here is the proposed time: Monday, 10:00-10:30

    Example 2:
    Question: Schedule a meeting for Alice and Bob for 1 hour on Tuesday. Alice is busy from 14:00-15:00 on Tuesday, Bob is free.
    Extracted Info: {{"participants": ["Alice", "Bob"], "duration": 60, "days": ["Tuesday"], "schedules": {{"Alice": ["14:00-15:00"], "Bob": []}}}}
    Reasoning: Alice is free except 14:00-15:00. Bob is free. So, 10:00-11:00 is an option.
    Proposed Time: Here is the proposed time: Tuesday, 10:00-11:00

    Considering the above, determine an appropriate meeting time given this extracted information and the question.
    Extracted Info: {extracted_info}
    Question: {question}

    Respond in the format 'Here is the proposed time: [day], [start_time]-[end_time]'
    Proposed Time:
    """
    return call_llm(prompt, system_instruction)

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response"""
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