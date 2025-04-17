import os
import json
import re
import math

def main(question):
    """
    Schedules meetings using a new approach: Multi-Agent Collaboration with Solution Generation & Verification.
    This script uses separate agents for extracting info, generating candidate solutions, and verifying them.
    This approach attempts to solve the limitations of previous iterations by explicitly generating solutions.
    """
    try:
        # 1. Information Extraction Agent: Extract relevant details.
        extracted_info = extract_meeting_info(question)
        if "Error" in extracted_info: return extracted_info

        # 2. Solution Generation Agent: Generate candidate schedules based on extracted information.
        candidate_schedules = generate_candidate_schedules(extracted_info)
        if "Error" in candidate_schedules: return candidate_schedules

        # 3. Verification Agent: Verify the generated schedules.
        verified_schedule = verify_schedules(question, candidate_schedules)
        if "Error" in verified_schedule: return verified_schedule

        return verified_schedule

    except Exception as e:
        return f"Error in main: {str(e)}"

def extract_meeting_info(question):
    """Extracts meeting information using LLM. Includes example with reasoning."""
    system_instruction = "You are an expert Information Extraction Agent. Focus on extracting accurate details."
    prompt = f"""
    Extract meeting details from the text. Include participants, duration, days, and constraints.
    Example:
    Input: Schedule a meeting for John and Jane for 30 minutes on Monday. John is busy 1-2pm. Jane prefers to meet before noon.
    Reasoning: Extract participants (John, Jane), duration (30), days (Monday), constraints (John busy 1-2pm, Jane prefers before noon).
    Output: {{"participants": ["John", "Jane"], "duration": 30, "days": ["Monday"], "constraints": ["John is busy 1-2pm", "Jane prefers before noon"]}}

    Now extract from: {question}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error extracting info: {str(e)}"

def generate_candidate_schedules(extracted_info):
    """Generates candidate schedules using LLM, incorporating extracted info. Includes example with reasoning."""
    system_instruction = "You are an expert Solution Generation Agent. Create valid meeting schedules."
    prompt = f"""
    Generate a candidate meeting schedule based on the extracted information. Consider all constraints and preferences.
    Example:
    Input: {{"participants": ["John", "Jane"], "duration": 30, "days": ["Monday"], "constraints": ["John is busy 1-2pm", "Jane prefers before noon"]}}
    Reasoning: Considering John's unavailability (1-2pm) and Jane's preference (before noon), a valid time would be 9:00-9:30.
    Output: Here is the proposed time: Monday, 9:00 - 9:30

    Now generate a schedule from: {extracted_info}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error generating candidate schedules: {str(e)}"

def verify_schedules(question, candidate_schedule):
    """Verifies the generated schedule using LLM. Includes example with reasoning."""
    system_instruction = "You are an expert Verification Agent. Ensure schedules meet all constraints."
    prompt = f"""
    Verify that the proposed meeting schedule is valid given the question.
    Example:
    Question: Schedule a meeting for John and Jane for 30 minutes on Monday. John is busy 1-2pm. Jane prefers to meet before noon.
    Schedule: Monday, 1:30 - 2:00
    Reasoning: John is busy during 1:30-2:00, so this schedule is invalid.
    Output: Invalid - John is busy.

    Question: {question}
    Schedule: {candidate_schedule}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error verifying schedule: {str(e)}"

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