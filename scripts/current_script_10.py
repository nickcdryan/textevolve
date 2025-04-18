import os
import json
import re
import datetime
from datetime import timedelta

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

def main(question):
    """Schedules meetings using multi-perspective analysis and iterative refinement."""
    try:
        # 1. Analyze the problem from multiple perspectives
        analysis_result = multi_perspective_analysis(question)

        # 2. Extract key information from the combined analysis
        meeting_info = extract_meeting_info(analysis_result["synthesis"])

        # 3. Find an available meeting slot based on extracted information
        available_slot = find_available_time_slot(meeting_info, question)
        return available_slot

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def multi_perspective_analysis(question):
    """Analyzes the question from different perspectives to extract relevant info."""
    system_instruction = "You are a multi-faceted analyst combining insights from different experts."
    prompt = f"""
    Analyze this meeting scheduling question from three perspectives:
    1. Information Extraction Expert: Focus on identifying key information (participants, duration, time, constraints).
    2. Constraint Satisfaction Expert: Focus on understanding the constraints and how they limit the solution space.
    3. Time Management Expert: Focus on time arithmetic and finding optimal meeting slots.

    Example:
    Question: Schedule a meeting for John and Jane for 30 minutes between 9:00 and 17:00 on Monday. John is busy 10:00-11:00, Jane is busy 13:00-14:00.
    Analysis:
    {{
        "information_extraction": "Participants: John, Jane; Duration: 30; Time: 9:00-17:00; Day: Monday; Constraints: John busy 10:00-11:00, Jane busy 13:00-14:00",
        "constraint_satisfaction": "Need to find a 30-minute slot that doesn't overlap with John's 10:00-11:00 or Jane's 13:00-14:00 on Monday",
        "time_management": "Possible slots: 9:00-9:30, 9:30-10:00, 11:00-11:30, 11:30-12:00... Check each slot for conflicts."
    }}

    Question: {question}
    Analysis:
    """
    analysis = call_llm(prompt, system_instruction)
    try:
        return json.loads(analysis)
    except:
        return {"synthesis": analysis}  # If JSON fails, return raw analysis

def extract_meeting_info(analysis):
    """Extracts key meeting information from the multi-perspective analysis."""
    system_instruction = "You are an expert at extracting meeting details from the analysis."
    prompt = f"""
    Based on this multi-perspective analysis, extract the key meeting information.
    Analysis: {analysis}

    Example:
    Analysis: {{"information_extraction": "Participants: John, Jane; Duration: 30; Time: 9:00-17:00; Day: Monday; Constraints: John busy 10:00-11:00, Jane busy 13:00-14:00", "constraint_satisfaction": "Need to find a 30-minute slot that doesn't overlap with John's 10:00-11:00 or Jane's 13:00-14:00 on Monday", "time_management": "Possible slots: 9:00-9:30, 9:30-10:00, 11:00-11:30, 11:30-12:00... Check each slot for conflicts."}}
    Output:
    {{
      "participants": ["John", "Jane"],
      "duration": 30,
      "days": ["Monday"],
      "work_hours": ["9:00", "17:00"],
      "schedules": {{
        "John": [["10:00", "11:00"]],
        "Jane": [["13:00", "14:00"]]
      }}
    }}

    Output:
    """
    extracted_info = call_llm(prompt, system_instruction)
    try:
        return json.loads(extracted_info)
    except:
        return "Error: Could not extract valid JSON"

def find_available_time_slot(info, question, max_attempts=5):
    """Finds an available time slot with iterative verification."""
    system_instruction = "You are an expert meeting scheduler."
    for attempt in range(max_attempts):
        proposal_prompt = f"""
        Given the meeting information: {info}, propose a valid meeting slot (day, start time - end time).
        Consider the constraints and schedules.

        Example:
        Info: {{"participants": ["John", "Jane"], "duration": 30, "days": ["Monday"], "work_hours": ["9:00", "17:00"], "schedules": {{"John": [["10:00", "11:00"]], "Jane": [["13:00", "14:00"]]}}}}
        Proposal: Monday, 9:00 - 9:30

        Proposal:
        """
        proposed_slot = call_llm(proposal_prompt, system_instruction)

        verifier_prompt = f"""
        Verify if this slot: {proposed_slot} is valid given the constraints: {info}.
        Problem: {question}
        Example:
        Slot: Monday, 9:00 - 9:30
        Info: {{"participants": ["John", "Jane"], "duration": 30, "days": ["Monday"], "work_hours": ["9:00", "17:00"], "schedules": {{"John": [["10:00", "11:00"]], "Jane": [["13:00", "14:00"]]}}}}
        Verification: Valid

        Verification:
        """
        verification_result = call_llm(verifier_prompt, system_instruction)

        if "Valid" in verification_result:
            return f"Here is the proposed time: {proposed_slot}"
        else:
            continue
    return "Error: Could not find a suitable meeting time"