import os
import re
import math

def main(question):
    """Main function to schedule meetings based on given constraints."""
    try:
        # Step 1: Extract structured information using LLM with multi-example prompting
        extracted_info = extract_meeting_info(question)

        # Step 2: Identify available time slots using LLM reasoning with verification
        available_slots = identify_available_time_slots(extracted_info)

        # Step 3: Propose a meeting time using LLM with multi-example prompting
        proposed_time = propose_meeting_time(available_slots, extracted_info)
        
        # Step 4: Final Verification
        final_verification = verify_final_solution(proposed_time, extracted_info, question)
        
        return final_verification

    except Exception as e:
        return f"Error processing the request: {str(e)}"

def extract_meeting_info(question):
    """Extract key information from the question using LLM with multi-example prompting."""
    system_instruction = "You are an expert at extracting meeting details."
    prompt = f"""
    Extract the following meeting details: participants, duration, days, and existing schedules.

    Example 1:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9-10, Mary is busy from 11-12.
    Extracted Info: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": ["9:00-10:00"], "Mary": ["11:00-12:00"]}}}}

    Example 2:
    Question: Schedule a meeting for Alice, Bob, and Charlie for 1 hour on Tuesday and Wednesday. Alice is busy from 14:00-15:00 on Tuesday, Bob is busy from 10:00-11:00 on Wednesday. Charlie is free.
    Extracted Info: {{"participants": ["Alice", "Bob", "Charlie"], "duration": 60, "days": ["Tuesday", "Wednesday"], "schedules": {{"Alice": ["14:00-15:00 (Tuesday)"], "Bob": ["10:00-11:00 (Wednesday)"], "Charlie": []}}}}

    Question: {question}
    Extracted Info:
    """
    return call_llm(prompt, system_instruction)

def identify_available_time_slots(extracted_info):
    """Identify available time slots based on extracted information using LLM."""
    system_instruction = "You are an expert at reasoning about schedules to find available time slots."
    prompt = f"""
    Based on these extracted meeting details, identify all available 30-minute time slots between 9:00 and 17:00.

    Extracted Info: {extracted_info}

    Available Time Slots:
    """
    return call_llm(prompt, system_instruction)

def propose_meeting_time(available_slots, extracted_info):
    """Propose a suitable meeting time based on available slots and participant constraints."""
    system_instruction = "You are skilled at proposing meeting times considering participant constraints."
    prompt = f"""
    Considering these available time slots and meeting details, propose the best meeting time.

    Available Time Slots: {available_slots}
    Meeting Details: {extracted_info}

    Example 1:
    Available Time Slots: Monday 10:00-10:30, Monday 14:00-14:30
    Meeting Details: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": [], "Mary": []}}}}
    Proposed Time: Monday, 10:00 - 10:30

    Proposed Time:
    """
    return call_llm(prompt, system_instruction)

def verify_final_solution(proposed_time, extracted_info, question):
    """Verify if the proposed time works with everyone's schedule and constraints."""
    system_instruction = "You are an expert verifier, who makes sure the proposed solution works with all the scheduling constraints."
    prompt = f"""
    Verify the proposed meeting time against all the given constraints and participant schedules.
    
    Question: {question}
    Proposed Time: {proposed_time}
    Extracted Info: {extracted_info}
    
    Example 1:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9-10, Mary is busy from 11-12.
    Proposed Time: Monday, 10:30-11:00
    Extracted Info: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": ["9:00-10:00"], "Mary": ["11:00-12:00"]}}}}
    Verification: The proposed time works for both John and Mary.
    
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