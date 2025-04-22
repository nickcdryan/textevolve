import os
import re
import math
import json

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

def extract_scheduling_information(question):
    """Extract scheduling information using LLM with examples."""
    system_instruction = "You are an expert at extracting scheduling information."
    prompt = f"""
    Extract the following information from the scheduling request: participants, duration, date(s), time range, existing schedules, and preferences.

    Example 1:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday between 9am and 5pm. John is busy from 10am to 11am and Mary is busy from 2pm to 3pm.
    Information: Participants: John, Mary; Duration: 30 minutes; Date: Monday; Time Range: 9am-5pm; John's Schedule: 10am-11am; Mary's Schedule: 2pm-3pm; Preferences: None

    Example 2:
    Question: You need to schedule a meeting for Nicholas, Sara, Helen, Brian, Nancy, Kelly and Judy for half an hour between the work hours of 9:00 to 17:00 on Monday. Nicholas is busy on Monday during 9:00 to 9:30, 11:00 to 11:30, 12:30 to 13:00, 15:30 to 16:00; Sara is busy on Monday during 10:00 to 10:30, 11:00 to 11:30; Helen is free the entire day. Brian is free the entire day. Nancy has blocked their calendar on Monday during 9:00 to 10:00, 11:00 to 14:00, 15:00 to 17:00; Kelly is busy on Monday during 10:00 to 11:30, 12:00 to 12:30, 13:30 to 14:00, 14:30 to 15:30, 16:30 to 17:00; Judy has blocked their calendar on Monday during 9:00 to 11:30, 12:00 to 12:30, 13:00 to 13:30, 14:30 to 17:00.
    Information: Participants: Nicholas, Sara, Helen, Brian, Nancy, Kelly, Judy; Duration: half an hour; Date: Monday; Time Range: 9:00-17:00; Nicholas's Schedule: 9:00-9:30, 11:00-11:30, 12:30-13:00, 15:30-16:00; Sara's Schedule: 10:00-10:30, 11:00-11:30; Helen's Schedule: Free; Brian's Schedule: Free; Nancy's Schedule: 9:00-10:00, 11:00-14:00, 15:00-17:00; Kelly's Schedule: 10:00-11:30, 12:00-12:30, 13:30-14:00, 14:30-15:30, 16:30-17:00; Judy's Schedule: 9:00-11:30, 12:00-12:30, 13:00-13:30, 14:30-17:00; Preferences: None

    Question: {question}
    Information:
    """
    return call_llm(prompt, system_instruction)

def find_best_time_slot(extracted_info):
    """Find the best available time slot based on extracted information."""
    system_instruction = "You are an expert at determining the best time slot for a meeting, considering all participants' schedules and preferences."
    prompt = f"""
    Given the following extracted scheduling information, determine the best available time slot.  Consider all participant schedules and any stated preferences to identify the optimal time. Return "No suitable time found" if no possibilities exist.

    Example 1:
    Information: Participants: John, Mary; Duration: 30 minutes; Date: Monday; Time Range: 9am-5pm; John's Schedule: 10am-11am; Mary's Schedule: 2pm-3pm; Preferences: None
    Best Time Slot: Monday, 9:00-9:30

    Example 2:
    Information: Participants: Nicholas, Sara; Duration: half an hour; Date: Monday; Time Range: 9:00-17:00; Nicholas's Schedule: 9:00-9:30, 11:00-11:30; Sara's Schedule: 10:00-10:30, 11:00-11:30; Preferences: None
    Best Time Slot: Monday, 9:30-10:00

    Information: {extracted_info}
    Best Time Slot:
    """
    return call_llm(prompt, system_instruction)

def verify_time_slot(question, proposed_time):
    """Verify that the proposed time slot is valid and adheres to all constraints."""
    system_instruction = "You are an expert at verifying time slots against scheduling constraints."
    prompt = f"""
    Verify that the proposed time slot is valid for the given scheduling request. Check for conflicts with participant schedules, adherence to time range, and satisfaction of any preferences. If there are any issues, explain the error. Otherwise, return VALID.

    Example 1:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday between 9am and 5pm. John is busy from 10am to 11am and Mary is busy from 2pm to 3pm. Proposed Time: Monday, 10:30-11:00
    Verification: VALID

    Example 2:
    Question: You need to schedule a meeting for Nicholas and Sara for half an hour between the work hours of 9:00 to 17:00 on Monday. Nicholas is busy on Monday during 9:00 to 9:30, 11:00 to 11:30; Sara is busy on Monday during 10:00 to 10:30, 11:00 to 11:30
    Proposed Time: Monday, 11:00-11:30
    Verification: Invalid, conflicts with Nicholas's and Sara's schedules.

    Question: {question}
    Proposed Time: {proposed_time}
    Verification:
    """
    return call_llm(prompt, system_instruction)

def format_answer(time_slot):
    """Format the answer in a consistent way."""
    return f"Here is the proposed time: {time_slot} "

def main(question):
    """Main function to schedule a meeting given the question."""
    try:
        # Extract scheduling information
        extracted_info = extract_scheduling_information(question)

        # Find the best time slot
        best_time = find_best_time_slot(extracted_info)

        # Verify the time slot
        verification = verify_time_slot(question, best_time)

        if "VALID" in verification:
            return format_answer(best_time)
        else:
            return "Error: " + verification

    except Exception as e:
        return f"Error: {str(e)}"