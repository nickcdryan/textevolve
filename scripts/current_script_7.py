import os
import json
import re
import math

def main(question):
    """
    Schedules meetings using a novel Chain-of-Verification approach:
    1. Extracts constraints and available times using LLM with embedded examples.
    2. Generates a PROPOSED solution using LLM.
    3. Verifies the solution with chained, specific checkers:
        - Temporal Constraint Verifier
        - Preference Constraint Verifier
        - Overall Solution Checker
    4. Returns verified solution.
    """
    try:
        # Extract constraints and available times
        extracted_data = extract_constraints_and_availability(question)
        if "Error" in extracted_data: return extracted_data

        # Generate a proposed solution
        proposed_solution = generate_proposed_solution(extracted_data)
        if "Error" in proposed_solution: return proposed_solution

        # Verify Temporal Constraints
        temporal_verification = verify_temporal_constraints(question, proposed_solution)
        if "Error" in temporal_verification or "Invalid" in temporal_verification: return temporal_verification

        # Verify Preference Constraints
        preference_verification = verify_preference_constraints(question, proposed_solution)
        if "Error" in preference_verification or "Invalid" in preference_verification: return preference_verification

        # Overall Solution Checker
        overall_check = overall_solution_check(question, proposed_solution)
        if "Error" in overall_check or "Invalid" in overall_check: return overall_check

        return proposed_solution

    except Exception as e:
        return f"Error in main: {str(e)}"

def extract_constraints_and_availability(question):
    """Extracts constraints and availability using LLM with examples."""
    system_instruction = "You are an information extraction expert."
    prompt = f"""
    Extract the following information from the input question: participants, duration, available days, time constraints, preferences.

    Example:
    Input: Schedule a meeting for John and Jane for 30 minutes on Monday. John is busy 1-2pm. Jane prefers to meet before noon.
    Output: {{"participants": ["John", "Jane"], "duration": 30, "days": ["Monday"], "busy_slots": {{"John": ["1:00-2:00"]}}, "preferences": {{"Jane": ["before noon"]}}}}

    Input: {question}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error extracting constraints: {str(e)}"

def generate_proposed_solution(extracted_data):
    """Generates a proposed meeting schedule using LLM."""
    system_instruction = "You are an expert at generating meeting schedules."
    prompt = f"""
    Given the extracted data, generate a proposed meeting schedule. Focus on the *earliest* possible valid time.

    Example:
    Input: {{"participants": ["John", "Jane"], "duration": 30, "days": ["Monday"], "busy_slots": {{"John": ["1:00-2:00"]}}, "preferences": {{"Jane": ["before noon"]}}}}
    Output: Here is the proposed time: Monday, 9:00 - 9:30

    Input: {extracted_data}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error generating proposed solution: {str(e)}"

def verify_temporal_constraints(question, proposed_solution):
    """Verifies temporal constraints using LLM."""
    system_instruction = "You are a strict temporal constraint checker."
    prompt = f"""
    Verify that the proposed solution satisfies all temporal constraints. Output 'Valid' if it does, otherwise output 'Invalid' and the reason.

    Example:
    Question: Schedule a meeting for John and Jane for 30 minutes on Monday. John is busy 1-2pm.
    Proposed Solution: Here is the proposed time: Monday, 1:30 - 2:00
    Output: Invalid - John is busy.

    Question: {question}
    Proposed Solution: {proposed_solution}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error verifying temporal constraints: {str(e)}"

def verify_preference_constraints(question, proposed_solution):
    """Verifies preference constraints using LLM."""
    system_instruction = "You are a strict preference constraint checker."
    prompt = f"""
    Verify that the proposed solution satisfies all preference constraints. Output 'Valid' if it does, otherwise output 'Invalid' and the reason.

    Example:
    Question: Schedule a meeting for John and Jane for 30 minutes on Monday. Jane prefers to meet before noon.
    Proposed Solution: Here is the proposed time: Monday, 2:00 - 2:30
    Output: Invalid - Jane prefers before noon.

    Question: {question}
    Proposed Solution: {proposed_solution}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error verifying preference constraints: {str(e)}"

def overall_solution_check(question, proposed_solution):
    """Verifies overall solution using LLM to catch any remaining errors."""
    system_instruction = "You are an expert at scheduling meetings and verifying solutions."
    prompt = f"""
    Carefully verify the proposed solution against the original question to make sure it is a VALID and COMPLETE answer.

    Example:
    Question: Schedule a meeting for John and Jane for 30 minutes on Monday. John is busy 1-2pm.
    Proposed Solution: Here is the proposed time: Monday, 9:00 - 9:30
    Output: Valid

    Question: {question}
    Proposed Solution: {proposed_solution}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error performing overall solution check: {str(e)}"

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