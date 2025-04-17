import os
import json
import re
import math

def main(question):
    """
    Schedules meetings using a novel approach: Iterative Constraint Satisfaction with Multi-Agent Verification.
    This script uses separate agents for initial constraint extraction, candidate schedule generation,
    and then *multiple* verification agents with different perspectives to iteratively refine and validate the solution.

    This approach differs fundamentally by using a dedicated constraint satisfaction loop.
    """
    try:
        # 1. Initial Constraint Extraction: Agent extracts constraints.
        extracted_constraints = extract_constraints(question)
        if "Error" in extracted_constraints: return extracted_constraints

        # 2. Candidate Schedule Generation: Agent generates a candidate schedule.
        candidate_schedule = generate_candidate_schedule(extracted_constraints)
        if "Error" in candidate_schedule: return candidate_schedule

        # 3. Iterative Constraint Satisfaction Loop
        verified_schedule = verify_schedule(question, candidate_schedule, extracted_constraints, max_attempts=3)
        if "Error" in verified_schedule: return verified_schedule

        return verified_schedule

    except Exception as e:
        return f"Error in main: {str(e)}"

def extract_constraints(question):
    """Extracts meeting constraints using LLM with a detailed example."""
    system_instruction = "You are an expert Constraint Extraction Agent. Focus on identifying all constraints and preferences."
    prompt = f"""
    Extract all scheduling constraints from the input question.
    Example:
    Input: Schedule a meeting for John and Jane for 30 minutes on Monday. John is busy 1-2pm. Jane prefers to meet before noon.
    Output: {{"participants": ["John", "Jane"], "duration": 30, "days": ["Monday"], "busy_slots": {{"John": ["1:00-2:00"]}}, "preferences": {{"Jane": ["before noon"]}}}}
    Now extract from: {question}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error extracting constraints: {str(e)}"

def generate_candidate_schedule(extracted_constraints):
    """Generates a candidate meeting schedule using LLM based on extracted constraints with example."""
    system_instruction = "You are a Candidate Schedule Generation Agent. Consider constraints to produce one valid meeting schedule."
    prompt = f"""
    Generate a candidate meeting schedule that adheres to the extracted constraints.
    Example:
    Input: {{"participants": ["John", "Jane"], "duration": 30, "days": ["Monday"], "busy_slots": {{"John": ["1:00-2:00"]}}, "preferences": {{"Jane": ["before noon"]}}}}
    Output: Here is the proposed time: Monday, 9:00 - 9:30
    Now generate from: {extracted_constraints}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error generating candidate schedule: {str(e)}"

def verify_schedule(question, candidate_schedule, extracted_constraints, max_attempts):
    """Verifies the generated schedule iteratively using multiple verification agents and a loop with example."""
    for attempt in range(max_attempts):
        # Multi-Agent Verification:
        # 1. Temporal Verifier: Checks for temporal conflicts.
        temporal_verification = verify_temporal_constraints(question, candidate_schedule, extracted_constraints)
        if "Error" in temporal_verification or "Invalid" in temporal_verification:
            candidate_schedule = generate_alternative_schedule(extracted_constraints, candidate_schedule, temporal_verification)
            continue

        # 2. Preference Verifier: Checks for preference violations.
        preference_verification = verify_preference_constraints(question, candidate_schedule, extracted_constraints)
        if "Error" in preference_verification or "Invalid" in preference_verification:
            candidate_schedule = generate_alternative_schedule(extracted_constraints, candidate_schedule, preference_verification)
            continue

        # If all verifications pass, return the schedule
        return candidate_schedule
    return "Could not find valid schedule after multiple attempts."

def verify_temporal_constraints(question, candidate_schedule, extracted_constraints):
    """Verifies temporal constraints using LLM with example."""
    system_instruction = "You are a Temporal Constraint Verification Agent. Validate temporal constraints and busy slots."
    prompt = f"""
    Verify that the candidate schedule does not violate any temporal constraints (e.g., busy slots).
    Example:
    Question: Schedule a meeting for John and Jane for 30 minutes on Monday. John is busy 1-2pm.
    Schedule: Monday, 1:30 - 2:00
    Output: Invalid - John is busy.
    Now verify: Question: {question}, Schedule: {candidate_schedule}, Constraints: {extracted_constraints}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error verifying temporal constraints: {str(e)}"

def verify_preference_constraints(question, candidate_schedule, extracted_constraints):
    """Verifies preference constraints using LLM with example."""
    system_instruction = "You are a Preference Constraint Verification Agent. Identify preference violations."
    prompt = f"""
    Check if the candidate schedule violates any preference constraints.
    Example:
    Question: Schedule a meeting for John and Jane for 30 minutes on Monday. Jane prefers to meet before noon.
    Schedule: Monday, 2:00 - 2:30
    Output: Invalid - Jane prefers before noon.
    Now verify: Question: {question}, Schedule: {candidate_schedule}, Constraints: {extracted_constraints}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error verifying preference constraints: {str(e)}"

def generate_alternative_schedule(extracted_constraints, current_schedule, feedback):
    """Generates an alternative schedule using LLM based on feedback with example."""
    system_instruction = "You are an Alternative Schedule Generation Agent. Provide a new valid schedule based on feedback."
    prompt = f"""
    Given the feedback and the current constraints, generate a new candidate meeting schedule.
    Example:
    Constraints: {{"participants": ["John", "Jane"], "duration": 30, "days": ["Monday"], "busy_slots": {{"John": ["1:00-2:00"]}}, "preferences": {{"Jane": ["before noon"]}}}}
    Current Schedule: Monday, 1:30 - 2:00
    Feedback: Invalid - John is busy.
    Output: Here is the proposed time: Monday, 9:00 - 9:30
    Now generate a schedule: Constraints: {extracted_constraints}, Current Schedule: {current_schedule}, Feedback: {feedback}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error generating alternative schedule: {str(e)}"

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response."""
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