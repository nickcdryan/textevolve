import os
import re
import math

def main(question):
    """Schedules meetings using a new LLM-driven approach with detailed extraction and constraint satisfaction."""
    try:
        # 1. Extract structured meeting data with verification
        meeting_data = extract_structured_data(question)
        if "Error" in meeting_data:
            return meeting_data

        # 2. Identify available time slots using LLM reasoning
        available_slots = find_available_time_slots(meeting_data, question)
        if "Error" in available_slots:
            return available_slots

        # 3. Propose meeting time based on available slots
        proposed_time = propose_meeting_time(available_slots, question)
        if "Error" in proposed_time:
            return proposed_time

        return proposed_time

    except Exception as e:
        return f"Error processing the request: {str(e)}"

def extract_structured_data(question, max_attempts=3):
    """Extracts structured meeting data (participants, duration, days, schedules) using LLM with examples."""
    system_instruction = "You are an expert at extracting structured meeting data from text."
    prompt = f"""
    You are an expert at extracting structured meeting data. Provide the data as a list of Python dictionaries.

    Example 1:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9:00-10:00, Mary is busy from 11:00-12:00.
    Data:
    [
     {{"participant": "John", "available_days": ["Monday"], "unavailable_times": ["9:00-10:00"]}},
     {{"participant": "Mary", "available_days": ["Monday"], "unavailable_times": ["11:00-12:00"]}},
     {{"meeting_duration": 30, "meeting_days": ["Monday"]}}
    ]

    Example 2:
    Question: Schedule a meeting for Alice, Bob, and Charlie for 1 hour on Tuesday and Wednesday. Alice is busy from 14:00-15:00 on Tuesday, Bob is busy from 10:00-11:00 on Wednesday. Charlie is free.
    Data:
    [
     {{"participant": "Alice", "available_days": ["Tuesday", "Wednesday"], "unavailable_times": ["14:00-15:00 (Tuesday)"]}},
     {{"participant": "Bob", "available_days": ["Tuesday", "Wednesday"], "unavailable_times": ["10:00-11:00 (Wednesday)"]}},
     {{"participant": "Charlie", "available_days": ["Tuesday", "Wednesday"], "unavailable_times": []}},
     {{"meeting_duration": 60, "meeting_days": ["Tuesday", "Wednesday"]}}
    ]

    Question: {question}
    Data:
    """
    extracted_data = call_llm(prompt, system_instruction)

    #Verification step
    try:
        # This is a check that tries to parse the output and return it. if there's an error we can try to refine it
        validated_extraction = validated_and_return_extraction(extracted_data, question, max_attempts)
        return validated_extraction
    except Exception as e:
        return f"Error in extraction: {str(e)}"
    
def validated_and_return_extraction(extracted_data, question, max_attempts):
    for attempt in range(max_attempts):
            validation_prompt = f"""
            You are an expert at verifying extracted information. You must use a Python list of dictionary format. Make sure it does NOT use JSON formats!
            Verify whether it follows the following rules
            1. Whether the values are valid and in the right data
            2. Whether the keys are properly set, and if there are keys that are missing
            3. Returns whether the structure is valid or invalid.

            Respond in a string format. 

            Example 1:
            Data:
            [
                {{"participant": "John", "available_days": ["Monday"], "unavailable_times": ["9:00-10:00"]}},
                {{"participant": "Mary", "available_days": ["Monday"], "unavailable_times": ["11:00-12:00"]}},
                {{"meeting_duration": 30, "meeting_days": ["Monday"]}}
            ]
            Validation:
                VALID

            Data: {extracted_data}
            Validation:
            """
            validation = call_llm(validation_prompt)
            if "VALID" in validation:
                return extracted_data
            else:
                print (f"Refining extraction, on attempt: {attempt}/{max_attempts}")
    return f"Data is invalid, after several attempts, failing with the question: {question}."

def find_available_time_slots(meeting_data, question, max_attempts=3):
    """Finds available time slots using LLM reasoning with few-shot examples."""
    system_instruction = "You are an expert at finding available time slots based on meeting data."
    prompt = f"""
    You are an expert at finding available time slots. 
    You are given participants, their availability, the duration, and the set days. 
    Derive the available times, and return that information as text. 

    Example:
    Data:
    [
     {{"participant": "John", "available_days": ["Monday"], "unavailable_times": ["9:00-10:00"]}},
     {{"participant": "Mary", "available_days": ["Monday"], "unavailable_times": ["11:00-12:00"]}},
     {{"meeting_duration": 30, "meeting_days": ["Monday"]}}
    ]
    Reasoning: John is busy 9:00-10:00, Mary is busy 11:00-12:00. A valid time is 10:00-10:30.
    Available slots: Monday, 10:00-10:30

    Data: {meeting_data}
    Reasoning:
    Available slots:
    """
    available_slots = call_llm(prompt, system_instruction)
    return available_slots

def propose_meeting_time(available_slots, question, max_attempts=3):
    """Proposes a meeting time using LLM."""
    system_instruction = "You are an expert at proposing the best meeting schedule time."
    prompt = f"""
    You are an expert at proposing a valid meeting time. 
    Here are the available times, and the question. You must only give me a time that works.

    Example:
    Available slots: Monday, 10:00-10:30
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday.
    Proposed time: Here is the proposed time: Monday, 10:00-10:30

    Available slots: {available_slots}
    Question: {question}
    Proposed time:
    """
    proposed_time = call_llm(prompt, system_instruction)
    return proposed_time

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