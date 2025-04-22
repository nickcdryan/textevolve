import os
import re
import math

def main(question):
    """Schedules meetings using a new LLM-driven approach that focuses on constraint extraction,
    structured schedule representation, and iterative time slot proposal with verification.

    HYPOTHESIS: By explicitly extracting constraints, representing schedules in a structured format,
    and iteratively proposing time slots with verification, we can significantly improve the accuracy
    and reliability of the generated meeting time.
    """
    try:
        # 1. Extract constraints and format information
        extracted_data = extract_constraints(question)
        if "Error" in extracted_data:
            return extracted_data

        # 2. Propose meeting time and check if the constraints are valid
        proposed_time = propose_meeting_time(extracted_data, question)
        if "Error" in proposed_time:
            return proposed_time

        return proposed_time

    except Exception as e:
        return f"Error processing the request: {str(e)}"

def extract_constraints(question):
    """Extracts meeting constraints, including participants, duration, days, schedules using LLM and validates format"""
    system_instruction = "You are an expert at extracting meeting details from text and formatting the extraction into structured data."
    prompt = f"""
    You are an expert at extracting the key data and formatting the data.
    You MUST respond with a JSON-like dictionary containing: participants, duration, days, schedules.

    Example 1:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9:00-10:00, Mary is busy from 11:00-12:00.
    Extraction:
    Participants: John, Mary
    Duration: 30 minutes
    Days: Monday
    John Schedule: 9:00-10:00
    Mary Schedule: 11:00-12:00
    

    Example 2:
    Question: Schedule a meeting for John, Jennifer, and Harold for a 1-hour meeting on Tuesday. Jennifer has meetings on Tuesday during 9:00 to 11:00 and on Wednesday from 13:00 to 15:00. Harold would rather not meet on Monday after 14:00.
    Extraction:
    Participants: John, Jennifer, Harold
    Duration: 1-hour
    Days: Tuesday
    Jennifer Schedule: 9:00 to 11:00 Tuesday, 13:00 to 15:00 Wednesday
    Harold Schedule: would rather not meet on Monday after 14:00
    

    Question: {question}
    Extraction:
    """
    extracted_data = call_llm(prompt, system_instruction)
    return extracted_data

def propose_meeting_time(extracted_data, question, max_attempts=3):
    """Proposes a meeting time and checks the constraints using LLM"""
    system_instruction = "You are an expert meeting scheduler, that analyzes extracted data and formulates responses, validating if it has the constraints necessary"
    prompt = f"""
    You are an expert at checking available times and extracting times
    
    Extracted Data: {extracted_data}
    Question: {question}
    Here are a couple of examples

    Example 1:
    Extracted Data:
    Participants: John, Mary
    Duration: 30 minutes
    Days: Monday
    John Schedule: 9:00-10:00
    Mary Schedule: 11:00-12:00
    Reasoning: Checking schedule time. Valid Solution is 10:00-10:30
    Proposed Time: Here is the proposed time: Monday, 10:00-10:30

    Example 2:
    Extracted Data:
    Participants: John, Jennifer, Harold
    Duration: 1-hour
    Days: Tuesday
    Jennifer Schedule: 9:00 to 11:00 Tuesday, 13:00 to 15:00 Wednesday
    Harold Schedule: would rather not meet on Monday after 14:00
    Reasoning: The time MUST be a Tuesday and NOT a Monday based on extraction
    Proposed Time: Here is the proposed time: Tuesday, 11:00-12:00

    Proposed time:
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