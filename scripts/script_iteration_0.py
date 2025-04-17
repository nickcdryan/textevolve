import os
import json
import re
import math

def main(question):
    """
    Orchestrates meeting scheduling using a multi-stage LLM-driven approach.
    Includes detailed information extraction, constraint analysis, and iterative refinement with a validator.
    """

    # 1. Extract information with examples using a Chain of Extraction
    extracted_info = extract_meeting_info(question)
    if "Error" in extracted_info:
        return extracted_info  # Propagate error

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
    """Extracts key meeting details (participants, duration, time constraints) using LLM."""
    system_instruction = "You are an expert at extracting structured information from text."
    prompt = f"""
    Extract the following information from the text: participants, duration, available days, time constraints.
    Example:
    Input: Schedule a meeting for John, Jane, and Peter for 1 hour on Monday or Tuesday between 9am and 5pm. John is busy from 10am-11am on Monday. Jane is unavailable from 2pm-3pm on Tuesday.
    Output:
    {{
      "participants": ["John", "Jane", "Peter"],
      "duration": "1 hour",
      "available_days": ["Monday", "Tuesday"],
      "time_constraints": "between 9am and 5pm. John is busy from 10am-11am on Monday. Jane is unavailable from 2pm-3pm on Tuesday."
    }}
    Now extract the same from:
    {question}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error extracting meeting info: {str(e)}"

def analyze_constraints(extracted_info):
    """Analyzes the extracted information to create constraints and preferences."""
    system_instruction = "You are an expert at analyzing constraints and preferences to guide meeting scheduling."
    prompt = f"""
    Analyze the extracted information and create a structured list of constraints and preferences.
    Example:
    Input:
    {{
      "participants": ["John", "Jane"],
      "duration": "30 minutes",
      "available_days": ["Monday"],
      "time_constraints": "between 9am and 5pm. John is busy from 10am-11am."
    }}
    Output:
    {{
      "duration": "30 minutes",
      "available_days": ["Monday"],
      "constraints": ["Meeting must be between 9am and 5pm", "John is busy from 10am-11am on Monday"]
    }}
    Now analyze the same from:
    {extracted_info}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error analyzing constraints: {str(e)}"

def generate_candidate_schedule(analyzed_constraints):
    """Generates a candidate schedule based on analyzed constraints."""
    system_instruction = "You are an expert meeting scheduler."
    prompt = f"""
    Generate a candidate meeting schedule.
    Example:
    Input:
    {{
      "duration": "30 minutes",
      "available_days": ["Monday"],
      "constraints": ["Meeting must be between 9am and 5pm", "John is busy from 10am-11am on Monday"]
    }}
    Output: Monday, 9:00 - 9:30
    Now generate the same from:
    {analyzed_constraints}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error generating candidate schedule: {str(e)}"

def verify_and_refine(candidate_schedule, extracted_info, question):
    """Verifies the solution and refines."""
    system_instruction = "You are a meeting scheduler expert. Double check the provided solution against constraints and context."
    prompt = f"""
    Given the context and schedule, verify that all hard constraints are met. If any constraints are violated, return an alternative, otherwise return the original.
    
    Example:
    Context:
    {{
      "participants": ["John", "Jane"],
      "duration": "30 minutes",
      "available_days": ["Monday"],
      "time_constraints": "between 9am and 5pm. John is busy from 10am-11am."
    }}
    
    Candidate schedule:
    Monday, 10:30 - 11:00
    
    Output: This is invalid, John is busy between 10:00 and 11:00. An alternative is Monday, 9:00 - 9:30
    
    
    Question:
    {question}
    
    Context:
    {extracted_info}
    
    Candidate schedule:
    {candidate_schedule}
    
    """
    
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error verifying and refining: {str(e)}"

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