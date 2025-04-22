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

def extract_scheduling_details(question):
    """Extract scheduling details from the question using a ReAct-like approach."""
    system_instruction = "You are an expert scheduling assistant."
    prompt = f"""
    You will extract scheduling details from a question in a step-by-step manner.

    Example 1:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday between 9am and 5pm. John is busy from 10am to 11am and Mary is busy from 2pm to 3pm.
    Step 1: Participants - John, Mary
    Step 2: Duration - 30 minutes
    Step 3: Date - Monday
    Step 4: Time Range - 9am to 5pm
    Step 5: John's Schedule - 10am to 11am
    Step 6: Mary's Schedule - 2pm to 3pm

    Example 2:
    Question: You need to schedule a meeting for Nicholas, Sara, Helen, Brian, Nancy, Kelly and Judy for half an hour between the work hours of 9:00 to 17:00 on Monday. Nicholas is busy on Monday during 9:00 to 9:30, 11:00 to 11:30, 12:30 to 13:00, 15:30 to 16:00; Sara is busy on Monday during 10:00 to 10:30, 11:00 to 11:30; Helen is free the entire day. Brian is free the entire day. Nancy has blocked their calendar on Monday during 9:00 to 10:00, 11:00 to 14:00, 15:00 to 17:00; Kelly is busy on Monday during 10:00 to 11:30, 12:00 to 12:30, 13:30 to 14:00, 14:30 to 15:30, 16:30 to 17:00; Judy has blocked their calendar on Monday during 9:00 to 11:30, 12:00 to 12:30, 13:00 to 13:30, 14:30 to 17:00.
    Step 1: Participants - Nicholas, Sara, Helen, Brian, Nancy, Kelly, Judy
    Step 2: Duration - half an hour
    Step 3: Date - Monday
    Step 4: Time Range - 9:00 to 17:00
    Step 5: Nicholas's Schedule - 9:00 to 9:30, 11:00 to 11:30, 12:30 to 13:00, 15:30 to 16:00
    Step 6: Sara's Schedule - 10:00 to 10:30, 11:00 to 11:30
    Step 7: Helen's Schedule - Free
    Step 8: Brian's Schedule - Free
    Step 9: Nancy's Schedule - 9:00 to 10:00, 11:00 to 14:00, 15:00 to 17:00
    Step 10: Kelly's Schedule - 10:00 to 11:30, 12:00 to 12:30, 13:30 to 14:00, 14:30 to 15:30, 16:30 to 17:00
    Step 11: Judy's Schedule - 9:00 to 11:30, 12:00 to 12:30, 13:00 to 13:30, 14:30 to 17:00

    Question: {question}
    """
    return call_llm(prompt, system_instruction)

def parse_extracted_details(extracted_text):
    """Parses the extracted details text and creates a structured dictionary."""
    details = {}
    for line in extracted_text.split('\n'):
        if ': ' in line:
            step, value = line.split(': ', 1)
            if "Participants" in step:
                details['participants'] = [p.strip() for p in value.split(',')]
            elif "Duration" in step:
                details['duration'] = value.strip()
            elif "Date" in step:
                details['date'] = value.strip()
            elif "Time Range" in step:
                details['time_range'] = value.strip()
            elif "'s Schedule" in step:
                name = step.split("'s Schedule")[0].strip()
                details[name] = [s.strip() for s in value.split(',')]
    return details

def find_available_time_slots_programmatic(details):
    """Find available time slots programmatically, given the extracted details."""
    # Placeholder for the programmatic logic.
    # Currently returns a string to indicate it needs implementation
    return "Programmatic time slot finding needs implementation"

def verify_time_slot_solution(question, proposed_time):
    """Verify if the time slot solution is valid against the original question."""
    system_instruction = "You are an expert at verifying time slot solutions based on scheduling requests."
    prompt = f"""
    Verify the proposed time slot solution for the given scheduling request. Identify any conflicts or issues.

    Example 1:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday between 9am and 5pm. John is busy from 10am to 11am and Mary is busy from 2pm to 3pm. Proposed Time: Monday, 11:00 - 11:30
    Verification: Valid. The proposed time does not conflict with John's or Mary's schedules.

    Example 2:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday between 9am and 5pm. John is busy from 10am to 11am and Mary is busy from 2pm to 3pm. Proposed Time: Monday, 2:30 - 3:00
    Verification: Invalid. The proposed time conflicts with Mary's schedule.

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
        # Extract scheduling details using the LLM.
        extracted_text = extract_scheduling_details(question)

        # Parse the extracted text into a structured dictionary.
        details = parse_extracted_details(extracted_text)

        #Find available time slots programmatically
        available_time = find_available_time_slots_programmatic(details)
        #Verify the output
        verification = verify_time_slot_solution(question, available_time)

        if "Invalid" not in verification:
            return format_answer(available_time)
        else:
            return "Error: " + verification

    except Exception as e:
        return f"Error: {str(e)}"