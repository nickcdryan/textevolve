import os
import re
import math

def main(question):
    """
    Schedules meetings using a structured approach with two specialized agents, multi-stage verification, and robust error handling.
    Leverages successful aspects of Iteration 5 with targeted improvements.
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
    """Extracts meeting details using a specialized extraction agent with multi-example prompting and verification."""
    system_instruction = "You are an expert at extracting meeting details from text. Your only job is to extract data, not to determine if the time works."

    for attempt in range(max_attempts):
        prompt = f"""
        You are an expert at extracting meeting details from text. Your goal is to pull out the important information. Your only job is to extract data, not to determine if the time works. Extract:
        - participants (list of names)
        - duration (integer, minutes)
        - days (list of strings, e.g., "Monday", "Tuesday")
        - existing schedules (dictionary, participant name -> list of time ranges "HH:MM-HH:MM (Day)")

        Example 1:
        Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9:00-10:00, Mary is busy from 11:00-12:00.
        Extraction: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": ["9:00-10:00 (Monday)"], "Mary": ["11:00-12:00 (Monday)"]}}}}

        Example 2:
        Question: Schedule a meeting for Alice, Bob, and Charlie for 1 hour on Tuesday and Wednesday. Alice is busy from 14:00-15:00 on Tuesday, Bob is busy from 10:00-11:00 on Wednesday. Charlie is free.
        Extraction: {{"participants": ["Alice", "Bob", "Charlie"], "duration": 60, "days": ["Tuesday", "Wednesday"], "schedules": {{"Alice": ["14:00-15:00 (Tuesday)"], "Bob": ["10:00-11:00 (Wednesday)"], "Charlie": []}}}}

        Example 3:
        Question: You need to schedule a meeting for Jonathan, Janice, Walter, Mary, Roger, Tyler and Arthur for half an hour between the work hours of 9:00 to 17:00 on Monday. Jonathan has meetings on Monday during 9:30 to 10:00, 12:30 to 13:30, 14:30 to 15:00; Janice has blocked their calendar on Monday during 9:00 to 9:30, 11:30 to 12:00, 12:30 to 13:30, 14:30 to 15:00, 16:00 to 16:30; Walter has blocked their calendar on Monday during 9:30 to 10:00, 11:30 to 12:00; Mary is busy on Monday during 12:00 to 12:30, 13:30 to 14:00; Roger has blocked their calendar on Monday during 9:30 to 10:30, 11:00 to 12:30, 13:00 to 13:30, 14:00 to 15:30, 16:00 to 16:30; Tyler has blocked their calendar on Monday during 9:30 to 11:00, 11:30 to 12:30, 13:30 to 14:00, 15:00 to 16:00; Arthur is busy on Monday during 10:00 to 11:30, 12:30 to 13:00, 13:30 to 14:00, 14:30 to 16:00;
        Extraction: {{"participants": ["Jonathan", "Janice", "Walter", "Mary", "Roger", "Tyler", "Arthur"], "duration": 30, "days": ["Monday"], "schedules": {{"Jonathan": ["9:30-10:00 (Monday)", "12:30-13:30 (Monday)", "14:30-15:00 (Monday)"], "Janice": ["9:00-9:30 (Monday)", "11:30-12:00 (Monday)", "12:30-13:30 (Monday)", "14:30-15:00 (Monday)", "16:00-16:30 (Monday)"], "Walter": ["9:30-10:00 (Monday)", "11:30-12:00 (Monday)"], "Mary": ["12:00-12:30 (Monday)", "13:30-14:00 (Monday)"], "Roger": ["9:30-10:30 (Monday)", "11:00-12:30 (Monday)", "13:00-13:30 (Monday)", "14:00-15:30 (Monday)", "16:00-16:30 (Monday)"], "Tyler": ["9:30-11:00 (Monday)", "11:30-12:30 (Monday)", "13:30-14:00 (Monday)", "15:00-16:00 (Monday)"], "Arthur": ["10:00-11:30 (Monday)", "12:30-13:00 (Monday)", "13:30-14:00 (Monday)", "14:30-16:00 (Monday)"]}}}}

        Question: {question}
        Extraction:
        """
        extracted_info = call_llm(prompt, system_instruction)

        # Validation step
        validation_prompt = f"""
        You are an expert at verifying extracted information. Given the question and the extraction, verify:
        1. Are all participants identified?
        2. Is the duration correct?
        3. Are all days mentioned included?
        4. Are the schedules correctly associated with each participant and day?

        If EVERYTHING is correct, respond EXACTLY with "VALID".
        Otherwise, explain the errors.

        Question: {question}
        Extracted Info (DO NOT LOAD AS JSON): {extracted_info}
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
    system_instruction = "You are an expert meeting scheduler. You are given all the information and must generate a final time that works."
    prompt = f"""
    You are an expert at scheduling meetings. Given the question and the extracted meeting details, your goal is to return a final proposed time in the format 'Here is the proposed time: [day], [start_time]-[end_time]'.
    You are given the following information:
    - Participants: list of names
    - Duration: integer, minutes
    - Days: list of strings, e.g., "Monday", "Tuesday"
    - Existing schedules: dictionary, participant name -> list of time ranges "HH:MM-HH:MM (Day)"

    Example 1:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9:00-10:00, Mary is busy from 11:00-12:00.
    Extraction: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": ["9:00-10:00 (Monday)"], "Mary": ["11:00-12:00 (Monday)"]}}}}
    Reasoning: John is available after 10:00 on Monday. Mary is available before 11:00 and after 12:00 on Monday. A 30-minute slot that works is 10:00-10:30.
    Proposed Time: Here is the proposed time: Monday, 10:00-10:30

    Example 2:
    Question: Schedule a meeting for Alice, Bob, and Charlie for 1 hour on Tuesday and Wednesday. Alice is busy from 14:00-15:00 on Tuesday, Bob is busy from 10:00-11:00 on Wednesday. Charlie is free.
    Extraction: {{"participants": ["Alice", "Bob", "Charlie"], "duration": 60, "days": ["Tuesday", "Wednesday"], "schedules": {{"Alice": ["14:00-15:00 (Tuesday)"], "Bob": ["10:00-11:00 (Wednesday)"], "Charlie": []}}}}
    Reasoning: On Tuesday, Alice is busy from 14:00-15:00. Bob and Charlie are free. A time that works for all is 10:00-11:00.
    Proposed Time: Here is the proposed time: Tuesday, 10:00-11:00

    Example 3:
    Question: You need to schedule a meeting for Noah, Ralph, Sean, John, Harold and Austin for half an hour between the work hours of 9:00 to 17:00 on Monday. Noah has blocked their calendar on Monday during 11:00 to 12:00, 12:30 to 13:00, 14:30 to 15:30, 16:30 to 17:00; Ralph has blocked their calendar on Monday during 10:30 to 11:00, 12:00 to 12:30, 13:00 to 14:00, 14:30 to 15:00, 16:30 to 17:00; Sean is busy on Monday during 13:00 to 13:30, 14:30 to 15:30, 16:30 to 17:00; John is busy on Monday during 9:30 to 10:30, 11:00 to 11:30, 13:00 to 16:00, 16:30 to 17:00; Harold is busy on Monday during 9:30 to 10:00, 11:30 to 12:30, 13:00 to 13:30, 14:00 to 15:30, 16:30 to 17:00; Austin has meetings on Monday during 10:00 to 11:00, 11:30 to 14:00, 14:30 to 17:00;
    Extraction: {{"participants": ["Noah", "Ralph", "Sean", "John", "Harold", "Austin"], "duration": 30, "days": ["Monday"], "schedules": {{"Noah": ["11:00-12:00 (Monday)", "12:30-13:00 (Monday)", "14:30-15:30 (Monday)", "16:30-17:00 (Monday)"], "Ralph": ["10:30-11:00 (Monday)", "12:00-12:30 (Monday)", "13:00-14:00 (Monday)", "14:30-15:00 (Monday)", "16:30-17:00 (Monday)"], "Sean": ["13:00-13:30 (Monday)", "14:30-15:30 (Monday)", "16:30-17:00 (Monday)"], "John": ["9:30-10:30 (Monday)", "11:00-11:30 (Monday)", "13:00-16:00 (Monday)", "16:30-17:00 (Monday)"], "Harold": ["9:30-10:00 (Monday)", "11:30-12:30 (Monday)", "13:00-13:30 (Monday)", "14:00-15:30 (Monday)", "16:30-17:00 (Monday)"], "Austin": ["10:00-11:00 (Monday)", "11:30-14:00 (Monday)", "14:30-17:00 (Monday)"]}}}}
    Reasoning: Analyzing the schedules, the only time slot available for all participants is 9:00-9:30.
    Proposed Time: Here is the proposed time: Monday, 9:00-9:30

    Considering the above, determine an appropriate meeting time given this extracted information and the question.
    Extracted Info: {extracted_info}
    Question: {question}

    Respond ONLY in the format 'Here is the proposed time: [day], [start_time]-[end_time]'
    Proposed Time:
    """
    return call_llm(prompt, system_instruction)