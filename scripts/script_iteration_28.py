import os
import re
import math

def main(question):
    """
    Schedules meetings using a structured approach with specialized agents and multi-stage verification.
    Leverages multi-example prompting for robust extraction and constraint satisfaction.
    """
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
    """Extracts meeting details using LLM with multi-example prompting and explicit verification."""
    system_instruction = "You are an expert at extracting meeting details."
    for attempt in range(max_attempts):
        prompt = f"""You are an expert at extracting meeting details from text. Your goal is to extract the following information accurately:
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
        Question: Schedule a meeting for Jonathan, Janice, Walter, Mary, Roger, Tyler and Arthur for half an hour between the work hours of 9:00 to 17:00 on Monday. Jonathan has meetings on Monday during 9:30 to 10:00, 12:30 to 13:30, 14:30 to 15:00; Janice has blocked their calendar on Monday during 9:00 to 9:30, 11:30 to 12:00, 12:30 to 13:30, 14:30 to 15:00, 16:00 to 16:30; Walter has blocked their calendar on Monday during 9:30 to 10:00, 11:30 to 12:00; Mary is busy on Monday during 12:00 to 12:30, 13:30 to 14:00; Roger has blocked their calendar on Monday during 9:30 to 10:30, 11:00 to 12:30, 13:00 to 13:30, 14:00 to 15:30, 16:00 to 16:30; Tyler has blocked their calendar on Monday during 9:30 to 11:00, 11:30 to 12:30, 13:30 to 14:00, 15:00 to 16:00; Arthur is busy on Monday during 10:00 to 11:30, 12:30 to 13:00, 13:30 to 14:00, 14:30 to 16:00;
        Extraction: {{"participants": ["Jonathan", "Janice", "Walter", "Mary", "Roger", "Tyler", "Arthur"], "duration": 30, "days": ["Monday"], "schedules": {{"Jonathan": ["9:30-10:00", "12:30-13:30", "14:30-15:00"], "Janice": ["9:00-9:30", "11:30-12:00", "12:30-13:30", "14:30-15:00", "16:00-16:30"], "Walter": ["9:30-10:00", "11:30-12:00"], "Mary": ["12:00-12:30", "13:30-14:00"], "Roger": ["9:30-10:30", "11:00-12:30", "13:00-13:30", "14:00-15:30", "16:00-16:30"], "Tyler": ["9:30-11:00", "11:30-12:30", "13:30-14:00", "15:00-16:00"], "Arthur": ["10:00-11:30", "12:30-13:00", "13:30-14:00", "14:30-16:00"]}}}}

        Question: {question}
        Extraction:"""
        extracted_info = call_llm(prompt, system_instruction)

        # Validation step - uses a separate LLM call for explicit verification.
        validation_prompt = f"""You are an expert at verifying extracted information for meeting scheduling. Given the question and the extracted information, your task is to verify the accuracy.
        Check the following:
        1. Are all participants correctly identified and listed?
        2. Is the meeting duration accurately extracted as a number (in minutes)?
        3. Are all the correct days considered and included?
        4. Are the schedules correctly associated with each participant, and are the time ranges valid?

        Example Input:
        Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9:00-10:00, Mary is busy from 11:00-12:00.
        Extracted Info: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": ["9:00-10:00"], "Mary": ["11:00-12:00"]}}}}
        Verification: VALID

        If EVERYTHING is correct and complete, respond EXACTLY with "VALID". Otherwise, provide a detailed explanation of all errors and omissions found.
        Question: {question}
        Extracted Info: {extracted_info}
        Verification:"""

        validation_result = call_llm(validation_prompt, system_instruction)
        if "VALID" in validation_result:
            return extracted_info
        else:
            print(f"Extraction validation failed (attempt {attempt+1}): {validation_result}")
    return f"Error: Extraction failed after multiple attempts: {validation_result}"

def schedule_meeting(extracted_info, question):
    """Schedules a meeting given extracted information. Includes validation."""
    system_instruction = "You are an expert meeting scheduler. Given all the information and constraints, generate a final time."
    prompt = f"""You are an expert at scheduling meetings. Given the question and extracted details, your goal is to return a final proposed time that satisfies all constraints.
    You are given:
    - Participants: list of names
    - Duration: integer, minutes
    - Days: list of strings, e.g., "Monday", "Tuesday"
    - Existing schedules: dictionary, participant name -> list of time ranges "HH:MM-HH:MM"

    Example 1:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9:00-10:00, Mary is busy from 11:00-12:00.
    Extraction: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": ["9:00-10:00"], "Mary": ["11:00-12:00"]}}}}
    Reasoning: John is available from 10:00. Mary is available from 9:00-11:00 and 12:00. A good time is 10:00-10:30.
    Proposed Time: Here is the proposed time: Monday, 10:00-10:30

    Example 2:
    Question: Schedule a meeting for Alice, Bob, and Charlie for 1 hour on Tuesday and Wednesday. Alice is busy from 14:00-15:00 on Tuesday, Bob is busy from 10:00-11:00 on Wednesday. Charlie is free.
    Extraction: {{"participants": ["Alice", "Bob", "Charlie"], "duration": 60, "days": ["Tuesday", "Wednesday"], "schedules": {{"Alice": ["14:00-15:00 (Tuesday)"], "Bob": ["10:00-11:00 (Wednesday)"], "Charlie": []}}}}
    Reasoning: On Tuesday, Alice is busy from 14:00-15:00 and Bob and Charlie are free. 10:00-11:00 works for all.
    Proposed Time: Here is the proposed time: Tuesday, 10:00-11:00

    Considering the above, find a meeting time given this extracted info and question.
    Extracted Info: {extracted_info}
    Question: {question}

    Respond in the format 'Here is the proposed time: [day], [start_time]-[end_time]'
    Proposed Time:"""
    proposed_time = call_llm(prompt, system_instruction)

    # Validation
    validation_prompt = f"""You are an expert at verifying meeting schedules. You are given the question, extracted info, and proposed time. Verify that the proposed time:
    1. Is in the format: 'Here is the proposed time: [day], [start_time]-[end_time]'
    2. Adheres to all schedules in 'Extracted Info'
    3. If the question specifies "earliest availability", return the earliest possible time.

    Example Input:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9:00-10:00, Mary is busy from 11:00-12:00. The group would like to meet at their earlist availability.
    Extracted Info: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": ["9:00-10:00"], "Mary": ["11:00-12:00"]}}}}
    Proposed Time: Here is the proposed time: Monday, 10:00-10:30
    Verification: VALID

    If EVERYTHING is correct and complete, respond EXACTLY with "VALID". Otherwise, provide a detailed explanation of all errors found.
    Question: {question}
    Extracted Info: {extracted_info}
    Proposed Time: {proposed_time}
    Verification:"""

    validation_result = call_llm(validation_prompt, system_instruction)
    if "VALID" in validation_result:
        return proposed_time
    else:
        return f"Error: Scheduling failed. {validation_result}"

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