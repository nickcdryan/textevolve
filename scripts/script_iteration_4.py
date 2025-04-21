import os
import json
import re
import math

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

def extract_and_verify_constraints(question, max_attempts=3):
    """Extracts constraints and verifies them using a multi-stage approach."""
    system_instruction = "You are an expert at extracting and verifying meeting constraints."
    # Use LLM to extract constraints.
    extraction_prompt = f"""
    Extract the participants, duration, time constraints, existing schedules, and preferences from the following text. Present it in a key-value format.

    Example:
    Text: You need to schedule a meeting for John and Jennifer for half an hour between 9:00 and 17:00 on Monday. John has no meetings. Jennifer has meetings on Monday from 9:00-11:00.
    Extraction:
    participants: John, Jennifer
    duration: 30 minutes
    constraints: between 9:00 and 17:00 on Monday
    schedules: John - None, Jennifer - Monday 9:00-11:00
    preferences: None

    Text: {question}
    Extraction:
    """
    extracted_info = call_llm(extraction_prompt, system_instruction)

    # Use LLM to verify the extracted constraints.
    verification_prompt = f"""
    Verify that the following extracted information contains all necessary components: participants, duration, constraints, schedules, and preferences. If something is missing or unclear, identify what it is.
    Extracted Information:
    {extracted_info}

    Example:
    Extracted Information:
    participants: John, Jennifer
    duration: 
    constraints: between 9:00 and 17:00 on Monday
    schedules: John - None, Jennifer - Monday 9:00-11:00
    preferences: None
    Feedback: Missing duration.

    Validation:
    """
    verification_result = call_llm(verification_prompt, system_instruction)

    #Retry logic to refine with feedback
    if "Missing" in verification_result:
      print(f"Retrying constraint extraction due to missing information: {verification_result}")
      refinement_prompt = f"""
        You extracted: {extracted_info}
        However, the following information is missing: {verification_result}
        Please extract all the constraints AGAIN:
        Text: {question}
      """
      extracted_info = call_llm(refinement_prompt, system_instruction)

    return extracted_info

def generate_and_verify_schedule(constraints, max_attempts=3):
    """Generates a meeting schedule and verifies that it satisfies the constraints."""
    system_instruction = "You are an expert meeting scheduler."

    # Use LLM to generate a meeting schedule.
    schedule_generation_prompt = f"""
    Given the extracted constraints, generate a meeting schedule.
    Constraints: {constraints}

    Example:
    Constraints:
    participants: John, Jennifer
    duration: 30 minutes
    constraints: between 9:00 and 17:00 on Monday
    schedules: John - None, Jennifer - Monday 9:00-11:00
    preferences: None
    Proposed Schedule: Here is the proposed time: Monday, 13:00 - 13:30

    Proposed Schedule:
    """
    proposed_schedule = call_llm(schedule_generation_prompt, system_instruction)

    # Use LLM to verify that the schedule satisfies all constraints.
    schedule_verification_prompt = f"""
    You are given the extracted constraints and a proposed meeting schedule. Verify if the schedule satisfies all the constraints. Respond with VALID or INVALID with the reason.

    Constraints: {constraints}
    Proposed Schedule: {proposed_schedule}

    Example:
    Constraints:
    participants: John, Jennifer
    duration: 30 minutes
    constraints: between 9:00 and 17:00 on Monday
    schedules: John - None, Jennifer - Monday 9:00-11:00
    preferences: None
    Proposed Schedule: Here is the proposed time: Monday, 13:00 - 13:30
    Verification: VALID

    Verification:
    """
    verification_result = call_llm(schedule_verification_prompt, system_instruction)

    return proposed_schedule if "VALID" in verification_result else f"INVALID: {verification_result}"

def main(question):
    """Main function to process the question and return the answer."""
    try:
        constraints = extract_and_verify_constraints(question)
        schedule = generate_and_verify_schedule(constraints)
        return schedule
    except Exception as e:
        return f"Error: {str(e)}"