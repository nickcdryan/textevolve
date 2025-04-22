import os
import re
import math

def main(question):
    """Schedules meetings using specialized agents and multi-stage verification."""
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
    """Extracts meeting details (participants, duration, days, schedules) using LLM."""
    system_instruction = "Expert at extracting meeting details."

    for attempt in range(max_attempts):
        prompt = f"""You are an expert at extracting meeting details. Extract:
        - participants (names)
        - duration (minutes)
        - days (e.g., "Monday")
        - existing schedules (participant -> time ranges "HH:MM-HH:MM")

        Example 1:
        Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy 9:00-10:00, Mary is busy 11:00-12:00.
        Extraction: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": ["9:00-10:00"], "Mary": ["11:00-12:00"]}}}}

        Example 2:
        Question: You need to schedule a meeting for Jonathan, Janice, Walter for half an hour on Monday. Jonathan has meetings 9:30-10:00, 12:30-13:30; Janice has blocked their calendar 9:00-9:30, 11:30-12:00; Walter blocked 9:30-10:00, 11:30-12:00.
        Extraction: {{"participants": ["Jonathan", "Janice", "Walter"], "duration": 30, "days": ["Monday"], "schedules": {{"Jonathan": ["9:30-10:00", "12:30-13:30"], "Janice": ["9:00-9:30", "11:30-12:00"], "Walter": ["9:30-10:00", "11:30-12:00"]}}}}

        Question: {question}
        Extraction:
        """
        extracted_info = call_llm(prompt, system_instruction)

        validation_prompt = f"""You are an expert at verifying extracted information.
        Given the question and the extracted information, check accuracy and completeness.
        1. Are all participants listed?
        2. Is the duration accurate?
        3. Are all the correct days included?
        4. Are the schedules correct?

        If EVERYTHING is correct, respond EXACTLY with "VALID". Otherwise, provide a detailed explanation of errors.

        Question: {question}
        Extracted Info: {extracted_info}
        Verification:
        """

        validation_result = call_llm(validation_prompt, system_instruction)
        if "VALID" in validation_result:
            return extracted_info
        else:
            print(f"Extraction failed (attempt {attempt+1}): {validation_result}")
    return f"Error: Extraction failed: {validation_result}"

def schedule_meeting(extracted_info, question):
    """Schedules a meeting given extracted information."""
    system_instruction = "Expert meeting scheduler. Generate a time that works."
    prompt = f"""You are an expert at scheduling meetings. Given the question and extracted details, return a final proposed time.
    You are given:
    - Participants: names
    - Duration: minutes
    - Days: strings
    - Existing schedules: participant -> time ranges "HH:MM-HH:MM"

    Example 1:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy 9:00-10:00, Mary is busy 11:00-12:00.
    Extraction: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": ["9:00-10:00"], "Mary": ["11:00-12:00"]}}}}
    Reasoning: John is available from 10:00. Mary is available 9:00-11:00 and 12:00. The best time is 10:00-10:30.
    Proposed Time: Here is the proposed time: Monday, 10:00-10:30

    Example 2:
    Question: You need to schedule a meeting for Nathan, Benjamin, Noah for half an hour on Monday. Nathan has no meetings. Benjamin is busy 10:00-10:30, 11:00-11:30; Noah is busy 9:30-13:30.
    Extraction: {{"participants": ["Nathan", "Benjamin", "Noah"], "duration": 30, "days": ["Monday"], "schedules": {{"Nathan": [], "Benjamin": ["10:00-10:30", "11:00-11:30"], "Noah": ["9:30-13:30"]}}}}
    Reasoning: Nathan is available all day. Benjamin is available 9:00-10:00, 10:30-11:00, 11:30-17:00. Noah is available 9:00-9:30, 13:30-17:00. With a duration of 30 minutes, a feasible time is 9:00-9:30.
    Proposed Time: Here is the proposed time: Monday, 9:00-9:30

    Considering the above, determine a meeting time given this extracted information and the question.
    Extracted Info: {extracted_info}
    Question: {question}

    Respond in the format 'Here is the proposed time: [day], [start_time]-[end_time]'
    Proposed Time:
    """
    proposed_time = call_llm(prompt, system_instruction)

    validation_prompt = f"""You are an expert at verifying meeting schedules. Verify:
    1. Is the proposed time in format: 'Here is the proposed time: [day], [start_time]-[end_time]'
    2. Does it adhere to all schedules in the 'Extracted Info'?

    If EVERYTHING is correct, respond EXACTLY with "VALID". Otherwise, provide a detailed explanation of errors.

    Question: {question}
    Extracted Info: {extracted_info}
    Proposed Time: {proposed_time}
    Verification:
    """

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