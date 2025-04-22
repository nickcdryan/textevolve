import os
import re
import math

def main(question):
    """
    Schedules meetings using a structured approach with specialized agents and multi-stage verification.
    """
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
    """Extracts meeting details with multi-example prompting and verification."""
    system_instruction = "You are an expert at extracting meeting details from text."

    for attempt in range(max_attempts):
        prompt = f"""
        You are an expert at extracting meeting details. Extract: participants, duration (minutes), days, schedules (participant: [time ranges]).

        Example 1:
        Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy 9:00-10:00, Mary 11:00-12:00.
        Extraction: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": ["9:00-10:00"], "Mary": ["11:00-12:00"]}}}}

        Example 2:
        Question: Schedule a meeting for Alice, Bob, and Charlie for 1 hour on Tuesday and Wednesday. Alice is busy 14:00-15:00 on Tuesday, Bob is busy 10:00-11:00 on Wednesday. Charlie is free.
        Extraction: {{"participants": ["Alice", "Bob", "Charlie"], "duration": 60, "days": ["Tuesday", "Wednesday"], "schedules": {{"Alice": ["14:00-15:00 (Tuesday)"], "Bob": ["10:00-11:00 (Wednesday)"], "Charlie": []}}}}

        Example 3:
        Question: Schedule Nathan, Benjamin, Noah, Bruce, and Matthew for 30 minutes on Monday. Nathan is free. Benjamin is busy 10:00-10:30, 11:00-11:30, 12:30-13:00, 14:00-14:30. Noah is busy 9:30-13:30, 14:00-14:30, 15:00-15:30, 16:00-17:00. Bruce is busy 9:30-10:30, 11:00-13:00, 13:30-14:00, 14:30-17:00. Matthew is busy 9:30-16:30.
        Extraction: {{"participants": ["Nathan", "Benjamin", "Noah", "Bruce", "Matthew"], "duration": 30, "days": ["Monday"], "schedules": {{"Nathan": [], "Benjamin": ["10:00-10:30", "11:00-11:30", "12:30-13:00", "14:00-14:30"], "Noah": ["9:30-13:30", "14:00-14:30", "15:00-15:30", "16:00-17:00"], "Bruce": ["9:30-10:30", "11:00-13:00", "13:30-14:00", "14:30-17:00"], "Matthew": ["9:30-16:30"]}}}}

        Question: {question}
        Extraction:
        """
        extracted_info = call_llm(prompt, system_instruction)

        validation_prompt = f"""
        You are an expert at verifying extracted meeting information. Check:
        1. Correct participants?
        2. Accurate duration (minutes)?
        3. Correct days?
        4. Schedules accurately reflect busy times for each participant on specified days?

        If EVERYTHING is correct, respond EXACTLY with "VALID".
        Otherwise, provide a DETAILED explanation of ALL errors/omissions.

        Question: {question}
        Extracted Info: {extracted_info}
        Verification:
        """

        validation_result = call_llm(validation_prompt, system_instruction)
        if "VALID" in validation_result:
            return extracted_info
        else:
            print(f"Extraction failed (attempt {attempt+1}): {validation_result}")
    return f"Error: Extraction failed after multiple attempts: {validation_result}"

def schedule_meeting(extracted_info, question):
    """Schedules a meeting using extracted information with dedicated validation."""
    system_instruction = "You are an expert meeting scheduler. Generate a final time that works."
    prompt = f"""
    You are an expert meeting scheduler. Return a proposed time satisfying all constraints.

    Example 1:
    Question: Schedule John and Mary for 30 minutes on Monday. John is busy 9:00-10:00, Mary 11:00-12:00.
    Extraction: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": ["9:00-10:00"], "Mary": ["11:00-12:00"]}}}}
    Reasoning: John is available after 10:00. Mary is available before 11:00 and after 12:00. So, 10:00-10:30 works.
    Proposed Time: Here is the proposed time: Monday, 10:00-10:30

    Example 2:
    Question: Schedule Alice, Bob, and Charlie for 1 hour on Tuesday and Wednesday. Alice is busy 14:00-15:00 on Tuesday, Bob is busy 10:00-11:00 on Wednesday. Charlie is free.
    Extraction: {{"participants": ["Alice", "Bob", "Charlie"], "duration": 60, "days": ["Tuesday", "Wednesday"], "schedules": {{"Alice": ["14:00-15:00 (Tuesday)"], "Bob": ["10:00-11:00 (Wednesday)"], "Charlie": []}}}}
    Reasoning: On Tuesday, Alice is busy 14:00-15:00, Bob/Charlie are free. A good time is 10:00-11:00.
    Proposed Time: Here is the proposed time: Tuesday, 10:00-11:00

    Example 3:
    Question: Schedule Nathan, Benjamin, Noah, Bruce, and Matthew for 30 minutes on Monday. Nathan is free. Benjamin is busy 10:00-10:30, 11:00-11:30, 12:30-13:00, 14:00-14:30. Noah is busy 9:30-13:30, 14:00-14:30, 15:00-15:30, 16:00-17:00. Bruce is busy 9:30-10:30, 11:00-13:00, 13:30-14:00, 14:30-17:00. Matthew is busy 9:30-16:30.
    Extraction: {{"participants": ["Nathan", "Benjamin", "Noah", "Bruce", "Matthew"], "duration": 30, "days": ["Monday"], "schedules": {{"Nathan": [], "Benjamin": ["10:00-10:30", "11:00-11:30", "12:30-13:00", "14:00-14:30"], "Noah": ["9:30-13:30", "14:00-14:30", "15:00-15:30", "16:00-17:00"], "Bruce": ["9:30-10:30", "11:00-13:00", "13:30-14:00", "14:30-17:00"], "Matthew": ["9:30-16:30"]}}}}
    Reasoning: Benjamin is available 9:00-10:00, 10:30-11:00, 11:30-12:30, 13:00-14:00, 14:30-17:00. Noah is available 9:00-9:30, 13:30-14:00, 14:30-15:00, 15:30-16:00. Bruce is available 9:00-9:30, 10:30-11:00, 13:00-13:30, 14:00-14:30. Matthew is available 9:00-9:30, 16:30-17:00. Therefore, 9:00-9:30 works.
    Proposed Time: Here is the proposed time: Monday, 9:00-9:30

    Considering the above, what's an appropriate meeting time given this extracted info and the question?
    Extracted Info: {extracted_info}
    Question: {question}

    Respond in the format 'Here is the proposed time: [day], [start_time]-[end_time]'
    Proposed Time:
    """
    proposed_time = call_llm(prompt, system_instruction)

    validation_prompt = f"""
    You are an expert at verifying meeting schedules. Verify:
    1. Proposed time is in the format 'Here is the proposed time: [day], [start_time]-[end_time]'?
    2. The proposed time adheres to ALL schedules in 'Extracted Info' (feasible for all)?

    If EVERYTHING is correct, respond EXACTLY with "VALID".
    Otherwise, provide a DETAILED explanation of ALL errors and omissions.

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