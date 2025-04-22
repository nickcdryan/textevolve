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
    """Extracts meeting details using a specialized extraction agent with multi-example prompting and verification."""
    system_instruction = "You are an expert at extracting meeting details from text."

    for attempt in range(max_attempts):
        prompt = f"""
        You are an expert at extracting meeting details from text. Your goal is to pull out the important information in a structured format.
        Extract:
        - participants (list of names)
        - duration (integer, minutes)
        - days (list of strings, e.g., "Monday", "Tuesday")
        - existing schedules (dictionary, participant name -> list of time ranges "HH:MM-HH:MM")
        - preferences (list of strings describing time/day preferences)

        Example 1:
        Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9:00-10:00, Mary is busy from 11:00-12:00.
        Extraction: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": ["9:00-10:00"], "Mary": ["11:00-12:00"]}}, "preferences": []}}

        Example 2:
        Question: Schedule a meeting for Alice, Bob, and Charlie for 1 hour on Tuesday and Wednesday. Alice is busy from 14:00-15:00 on Tuesday, Bob is busy from 10:00-11:00 on Wednesday. Charlie is free.
        Extraction: {{"participants": ["Alice", "Bob", "Charlie"], "duration": 60, "days": ["Tuesday", "Wednesday"], "schedules": {{"Alice": ["14:00-15:00 (Tuesday)"], "Bob": ["10:00-11:00 (Wednesday)"], "Charlie": []}}, "preferences": []}}

        Example 3:
        Question: Schedule a meeting for Janice and Walter for half an hour between 9:00 and 17:00 on Monday. Janice has blocked their calendar on Monday during 9:00 to 9:30, 11:30 to 12:00. Walter is busy on Monday during 9:30 to 10:00, 11:30 to 12:00. Janice would prefer to meet before noon.
        Extraction: {{"participants": ["Janice", "Walter"], "duration": 30, "days": ["Monday"], "schedules": {{"Janice": ["9:00-9:30", "11:30-12:00"], "Walter": ["9:30-10:00", "11:30-12:00"]}}, "preferences": ["Janice would prefer to meet before noon."]}}

        Question: {question}
        Extraction:
        """
        extracted_info = call_llm(prompt, system_instruction)

        # Validation step: Check that all key information is extracted and well-formatted
        validation_prompt = f"""
        You are an expert at verifying extracted information. Given the question and the extraction, verify:
        1. Are all participants identified?
        2. Is the duration correct?
        3. Are all days mentioned included?
        4. Are the schedules correctly associated with each participant and day?
        5. Are all preferences correctly extracted?

        If EVERYTHING is correct, respond EXACTLY with "VALID".
        Otherwise, explain the errors clearly.

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
    """Schedules a meeting given extracted information, generating the time using LLM."""
    system_instruction = "You are an expert meeting scheduler. You are given all the information and must generate a final proposed time that works."
    prompt = f"""
    You are an expert at scheduling meetings. Given the question and the extracted meeting details, your goal is to return a final proposed time that works for everyone.
    You are given the following information:
    - Participants: list of names
    - Duration: integer, minutes
    - Days: list of strings, e.g., "Monday", "Tuesday"
    - Existing schedules: dictionary, participant name -> list of time ranges "HH:MM-HH:MM"
    - Preferences: a list of preference strings

    Example 1:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9:00-10:00, Mary is busy from 11:00-12:00.
    Extraction: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": ["9:00-10:00"], "Mary": ["11:00-12:00"]}}, "preferences": []}}
    Reasoning: John is available from 10:00 onward. Mary is available from 9:00-11:00 and 12:00 onward. The best available time that works for both is 10:00-10:30.
    Proposed Time: Here is the proposed time: Monday, 10:00-10:30

    Example 2:
    Question: Schedule a meeting for Alice, Bob, and Charlie for 1 hour on Tuesday and Wednesday. Alice is busy from 14:00-15:00 on Tuesday, Bob is busy from 10:00-11:00 on Wednesday. Charlie is free.
    Extraction: {{"participants": ["Alice", "Bob", "Charlie"], "duration": 60, "days": ["Tuesday", "Wednesday"], "schedules": {{"Alice": ["14:00-15:00 (Tuesday)"], "Bob": ["10:00-11:00 (Wednesday)"], "Charlie": []}}, "preferences": []}}
    Reasoning: On Tuesday, Alice is busy from 14:00-15:00 and Bob and Charlie are free. A time that works for all is 10:00-11:00.
    Proposed Time: Here is the proposed time: Tuesday, 10:00-11:00

    Example 3:
    Question: Schedule a meeting for Janice and Walter for half an hour between 9:00 and 17:00 on Monday. Janice has blocked their calendar on Monday during 9:00 to 9:30, 11:30 to 12:00. Walter is busy on Monday during 9:30 to 10:00, 11:30 to 12:00. Janice would prefer to meet before noon.
    Extraction: {{"participants": ["Janice", "Walter"], "duration": 30, "days": ["Monday"], "schedules": {{"Janice": ["9:00-9:30", "11:30-12:00"], "Walter": ["9:30-10:00", "11:30-12:00"]}}, "preferences": ["Janice would prefer to meet before noon."]}}
    Reasoning: Analyzing the schedules, both Janice and Walter are free from 10:00-11:30 and from 12:00-17:00. Considering Janice's preference to meet before noon, the best time is 10:00-10:30.
    Proposed Time: Here is the proposed time: Monday, 10:00-10:30
        
    Considering the above, analyze the schedules, preferences and propose a meeting time.
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