import os
import json
import re
import math

def main(question):
    """
    Orchestrates meeting scheduling using a multi-stage LLM-driven approach with enhanced extraction and verification.
    """

    # 1. Extract information with examples and verification loop
    extracted_info = extract_meeting_info(question)
    if "Error" in extracted_info:
        return extracted_info

    # 2. Analyze constraints and preferences
    analyzed_constraints = analyze_constraints(extracted_info)
    if "Error" in analyzed_constraints:
        return analyzed_constraints

    # 3. Generate a candidate schedule and validate
    candidate_schedule = generate_candidate_schedule(analyzed_constraints)
    if "Error" in candidate_schedule:
        return candidate_schedule

    # 4. Verify and refine the solution
    verified_solution = verify_and_refine(candidate_schedule, extracted_info, question)
    return verified_solution

def extract_meeting_info(question):
    """Extracts key meeting details (participants, duration, time constraints) using LLM with example."""
    system_instruction = "You are an expert at extracting structured information from text, focus on meeting details."
    prompt = f"""
    Extract the following information from the text: participants, duration, available days, time constraints. Include preferred or avoided times.

    Example:
    Input: Schedule a meeting for John and Jane for 30 minutes on Monday. John is busy 1-2pm. Jane prefers to meet before noon.
    Output:
    {{
      "participants": ["John", "Jane"],
      "duration": "30 minutes",
      "available_days": ["Monday"],
      "time_constraints": "John is busy 1-2pm. Jane prefers to meet before noon."
    }}

    Now extract from:
    {question}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error extracting meeting info: {str(e)}"

def analyze_constraints(extracted_info):
    """Analyzes the extracted information to create constraints and preferences with example."""
    system_instruction = "You are an expert at analyzing constraints and preferences for scheduling."
    prompt = f"""
    Analyze the extracted information to create structured constraints.

    Example:
    Input:
    {{
      "participants": ["John", "Jane"],
      "duration": "30 minutes",
      "available_days": ["Monday"],
      "time_constraints": "John is busy 1-2pm. Jane prefers to meet before noon."
    }}
    Output:
    {{
      "duration": "30 minutes",
      "available_days": ["Monday"],
      "constraints": ["John is unavailable 1-2pm"],
      "preferences": ["Jane prefers before noon"]
    }}
    Now analyze from:
    {extracted_info}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error analyzing constraints: {str(e)}"

def generate_candidate_schedule(analyzed_constraints):
    """Generates a candidate schedule based on analyzed constraints with example."""
    system_instruction = "You are an expert meeting scheduler, skilled at creating conflict-free schedules."
    prompt = f"""
    Generate a candidate meeting schedule that satisfies constraints.

    Example:
    Input:
    {{
      "duration": "30 minutes",
      "available_days": ["Monday"],
      "constraints": ["John is unavailable 1-2pm"],
      "preferences": ["Jane prefers before noon"]
    }}
    Output: Monday, 9:00 - 9:30

    Now generate from:
    {analyzed_constraints}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error generating candidate schedule: {str(e)}"

def verify_and_refine(candidate_schedule, extracted_info, question):
    """Verifies the solution and refines if necessary with example."""
    system_instruction = "You are a meeting scheduler expert. Double check and refine the provided solution against constraints."
    prompt = f"""
    Verify that the candidate schedule meets all hard constraints. If violated, return an alternative.

    Example:
    Context:
    {{
      "participants": ["John", "Jane"],
      "duration": "30 minutes",
      "available_days": ["Monday"],
      "time_constraints": "John is busy 1-2pm. Jane prefers to meet before noon."
    }}

    Candidate schedule: Monday, 1:30 - 2:00

    Output: Invalid. John is busy. Alternative: Monday, 9:00 - 9:30

    Question: {question}
    Context: {extracted_info}
    Candidate schedule: {candidate_schedule}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error verifying and refining: {str(e)}"

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