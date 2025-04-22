import os
import re
import math

def main(question):
    """Schedules meetings by translating the question into structured data, and use the LLM to find the best possible time.
    This approach prioritizes LLM-driven structure generation, with minimal hardcoded logic.
    """
    try:
        structured_data = translate_to_structure(question)
        if "Error" in structured_data:
            return structured_data

        proposed_time = schedule_meeting(structured_data, question)
        if "Error" in proposed_time:
            return proposed_time

        return proposed_time

    except Exception as e:
        return f"Error processing the request: {str(e)}"

def translate_to_structure(question):
    """Translates the meeting request question into a structured format."""
    system_instruction = "You are expert at translating text to structured data"
    prompt = f"""You are an expert at translating meeting scheduling questions into structured data format, focusing on capturing key information.

    Example 1:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy 9:00-10:00, Mary is busy 11:00-12:00.
    Structured: participants: John, Mary; duration: 30; days: Monday; John_busy: 9:00-10:00; Mary_busy: 11:00-12:00

    Example 2:
    Question: You need to schedule a meeting for Jonathan, Janice, Walter for half an hour on Monday. Jonathan has meetings 9:30-10:00, 12:30-13:30; Janice has blocked their calendar 9:00-9:30, 11:30-12:00; Walter blocked 9:30-10:00, 11:30-12:00.
    Structured: participants: Jonathan, Janice, Walter; duration: 30; days: Monday; Jonathan_busy: 9:30-10:00, 12:30-13:30; Janice_busy: 9:00-9:30, 11:30-12:00; Walter_busy: 9:30-10:00, 11:30-12:00

    Question: {question}
    Structured:
    """
    structured_data = call_llm(prompt, system_instruction)
    return structured_data

def schedule_meeting(structured_data, question):
    """Schedules a meeting given structured data and original question context, finding best possibility."""
    system_instruction = "You are an expert meeting scheduler. Generate a valid proposed time that satisfies all constraints."
    prompt = f"""You are an expert at scheduling meetings. You will receive a structured data and question and must output proposed meeting time.

    Example 1:
    Structured: participants: John, Mary; duration: 30; days: Monday; John_busy: 9:00-10:00; Mary_busy: 11:00-12:00
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy 9:00-10:00, Mary is busy 11:00-12:00.
    Reasoning: John is available from 10:00. Mary is available 9:00-11:00 and 12:00. The best time is 10:00-10:30.
    Proposed Time: Here is the proposed time: Monday, 10:00-10:30

    Example 2:
    Structured: participants: Jonathan, Janice, Walter; duration: 30; days: Monday; Jonathan_busy: 9:30-10:00, 12:30-13:30; Janice_busy: 9:00-9:30, 11:30-12:00; Walter_busy: 9:30-10:00, 11:30-12:00
    Question: You need to schedule a meeting for Jonathan, Janice, Walter for half an hour on Monday. Jonathan has meetings 9:30-10:00, 12:30-13:30; Janice has blocked their calendar 9:00-9:30, 11:30-12:00; Walter blocked 9:30-10:00, 11:30-12:00.
    Reasoning: All are available from 13:30. Proposed Time is 13:30 to 14:00.
    Proposed Time: Here is the proposed time: Monday, 13:30-14:00

    Structured: {structured_data}
    Question: {question}
    Reasoning:
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