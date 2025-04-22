import os
import re
import math

def main(question):
    """Schedules meetings using a structured extraction, reasoning, and generation approach with validation.

    This iteration introduces a structured reasoning approach where the LLM is explicitly guided through steps,
    and multiple validation points are introduced in the extraction process for better results and insights.

    HYPOTHESIS: Explicitly guiding the LLM through a structured reasoning process, combined with validations,
    will improve scheduling accuracy and provide insights into failure points.
    """
    try:
        # Step 1: Extract structured info using LLM with validation
        extracted_info = extract_meeting_info_with_validation(question)
        if "Error" in extracted_info:
            return extracted_info

        # Step 2: Identify available time slots using LLM with structured reasoning
        available_slots = identify_available_time_slots(extracted_info, question)

        # Step 3: Propose a meeting time using LLM and the analyzed data
        proposed_time = propose_meeting_time(available_slots, extracted_info, question)

        # Step 4: Validate the final proposed time for hard constraints
        final_verification = verify_final_solution(proposed_time, extracted_info, question)

        return final_verification

    except Exception as e:
        return f"Error processing the request: {str(e)}"

def extract_meeting_info_with_validation(question, max_attempts=3):
    """Extracts structured information from the question using LLM with multi-example prompting and validation."""
    system_instruction = "You are an expert at extracting and validating meeting details."
    for attempt in range(max_attempts):
        prompt = f"""
        You are an expert at extracting meeting details from text. Extract:
        - participants (list of names)
        - duration (integer, minutes)
        - days (list of strings, e.g., "Monday", "Tuesday")
        - existing schedules (dictionary, participant name -> list of time ranges "HH:MM-HH:MM")

        Example 1:
        Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9:00-10:00, Mary is busy from 11:00-12:00.
        Extraction: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": ["09:00-10:00"], "Mary": ["11:00-12:00"]}}}}

        Example 2:
        Question: Schedule a meeting for Alice, Bob, and Charlie for 1 hour on Tuesday and Wednesday. Alice is busy from 14:00-15:00 on Tuesday, Bob is busy from 10:00-11:00 on Wednesday. Charlie is free.
        Extraction: {{"participants": ["Alice", "Bob", "Charlie"], "duration": 60, "days": ["Tuesday", "Wednesday"], "schedules": {{"Alice": ["14:00-15:00 (Tuesday)"], "Bob": ["10:00-11:00 (Wednesday)"], "Charlie": []}}}}

        Question: {question}
        Extraction:
        """
        extracted_info = call_llm(prompt, system_instruction)

        # Validation step
        validation_prompt = f"""
        You are an expert at verifying extracted information. Given the question and the extraction, verify:
        1.  Are all participants identified?
        2.  Is the duration correct?
        3.  Are all days mentioned included?
        4.  Are the schedules correctly associated with each participant and day?

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

def identify_available_time_slots(extracted_info, question):
    """Identify available time slots based on extracted information using LLM."""
    system_instruction = "You are an expert at reasoning about schedules to find available time slots."
    prompt = f"""
    You are an expert schedule analyzer. Using the meeting details and participant schedules provided, follow these steps:
    1. List all participants.
    2. For each day, list all possible 30-minute time slots between 9:00 and 17:00.
    3. Eliminate time slots where any participant has a conflict, based on their schedules.
    4.  Return the complete list of AVAILABLE time slots in the form:
        [Day, HH:MM-HH:MM], [Day, HH:MM-HH:MM], ...

    Example:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9-10, Mary is busy from 11-12.
    Extracted Info: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": ["09:00-10:00"], "Mary": ["11:00-12:00"]}}}}
    Available Time Slots: [Monday, 10:00-10:30], [Monday, 10:30-11:00], [Monday, 12:00-12:30], ..., [Monday, 16:30-17:00]

    Question: {question}
    Extracted Info: {extracted_info}
    Available Time Slots:
    """
    return call_llm(prompt, system_instruction)

def propose_meeting_time(available_slots, extracted_info, question):
    """Propose a suitable meeting time based on available slots and participant constraints."""
    system_instruction = "You are skilled at proposing meeting times considering participant constraints."
    prompt = f"""
    You are an expert meeting scheduler. Given the available time slots and meeting details, propose the BEST meeting time. Respond in the format:
    Here is the proposed time: [Day], [Start Time]-[End Time]

    Example:
    Available Time Slots: [Monday, 10:00-10:30], [Monday, 14:00-14:30]
    Meeting Details: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"]}}
    Proposed Time: Here is the proposed time: Monday, 10:00-10:30

    Available Time Slots: {available_slots}
    Meeting Details: {extracted_info}
    Proposed Time:
    """
    return call_llm(prompt, system_instruction)

def verify_final_solution(proposed_time, extracted_info, question):
    """Verify if the proposed time works with everyone's schedule and constraints."""
    system_instruction = "You are an expert verifier."
    prompt = f"""
    You are an expert meeting scheduler. Verify that the proposed meeting time works for ALL participants.
    Based on the schedules extracted, confirm the time does not conflict.
    Respond EXACTLY with "VALID" if the proposed time works. Otherwise, explain the conflict.

    Example:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9-10, Mary is busy from 11-12.
    Proposed Time: Here is the proposed time: Monday, 10:30-11:00
    Extracted Info: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": ["09:00-10:00"], "Mary": ["11:00-12:00"]}}}}
    Verification: VALID

    Question: {question}
    Proposed Time: {proposed_time}
    Extracted Info: {extracted_info}
    Verification:
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