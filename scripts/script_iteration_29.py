import os
import re
import math

def main(question):
    """
    Schedules meetings using a new approach that focuses on structured representation and iterative constraint satisfaction.
    HYPOTHESIS: Explicitly representing and tracking constraints throughout the process will improve constraint satisfaction.
    """
    try:
        # Step 1: Extract constraints and initial information
        extracted_data = extract_constraints(question)
        if "Error" in extracted_data:
            return extracted_data

        # Step 2: Propose a potential meeting time
        proposed_time = propose_meeting_time(extracted_data, question)
        if "Error" in proposed_time:
            return proposed_time

        # Step 3: Verify the proposed time against all constraints
        verification_result = verify_meeting_time(proposed_time, extracted_data, question)
        if "Error" in verification_result:
            return verification_result

        return proposed_time

    except Exception as e:
        return f"Error processing the request: {str(e)}"

def extract_constraints(question):
    """Extracts key constraints and information using an LLM."""
    system_instruction = "You are an expert at extracting constraints from meeting scheduling requests."
    prompt = f"""You are an expert at extracting constraints for meeting scheduling. Extract:
    - participants (names)
    - duration (minutes)
    - days (e.g., "Monday")
    - schedules (participant -> time ranges "HH:MM-HH:MM")
    - preferences (participant -> e.g., "avoid Monday")

    Example 1:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy 9:00-10:00, Mary is busy 11:00-12:00.
    Extraction: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": ["9:00-10:00"], "Mary": ["11:00-12:00"]}}, "preferences": {{}}}}

    Example 2:
    Question: Schedule a meeting for Alice, Bob for 1 hour on Tuesday. Alice is busy 14:00-15:00, Bob would rather not meet after 16:00.
    Extraction: {{"participants": ["Alice", "Bob"], "duration": 60, "days": ["Tuesday"], "schedules": {{"Alice": ["14:00-15:00"], "Bob": []}}, "preferences": {{"Bob": "avoid after 16:00"}}}}

    Question: {question}
    Extraction:
    """
    extracted_data = call_llm(prompt, system_instruction)
    return extracted_data

def propose_meeting_time(extracted_data, question):
    """Proposes a potential meeting time based on extracted data."""
    system_instruction = "You are an expert at proposing valid meeting times."
    prompt = f"""Given the extracted data, propose a potential meeting time.

    Example:
    Extracted: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": ["9:00-10:00"], "Mary": ["11:00-12:00"]}}, "preferences": {{}}}}
    Proposed Time: Here is the proposed time: Monday, 10:00-10:30

    Extracted: {extracted_data}
    Proposed Time:
    """
    proposed_time = call_llm(prompt, system_instruction)
    return proposed_time

def verify_meeting_time(proposed_time, extracted_data, question):
    """Verifies the proposed meeting time against all constraints."""
    system_instruction = "You are an expert at verifying meeting times against all constraints."
    prompt = f"""Verify if the proposed meeting time satisfies all constraints.
    Extracted Data: {extracted_data}
    Proposed Time: {proposed_time}

    Example 1:
    Extracted: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": ["9:00-10:00"], "Mary": ["11:00-12:00"]}}, "preferences": {{}}}}
    Proposed Time: Here is the proposed time: Monday, 10:00-10:30
    Verification: VALID

    Example 2:
    Extracted: {{"participants": ["Alice", "Bob"], "duration": 60, "days": ["Tuesday"], "schedules": {{"Alice": ["14:00-15:00"], "Bob": []}}, "preferences": {{"Bob": "avoid after 16:00"}}}}
    Proposed Time: Here is the proposed time: Tuesday, 16:30-17:30
    Verification: INVALID. Bob prefers to avoid meetings after 16:00.

    Extracted Data: {extracted_data}
    Proposed Time: {proposed_time}
    Verification:
    """
    verification_result = call_llm(prompt, system_instruction)

    if "INVALID" in verification_result:
        return f"Error: {verification_result}"
    return verification_result

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