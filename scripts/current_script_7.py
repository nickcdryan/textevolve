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
    """Schedules meetings using a new approach: Multi-stage information extraction with structured output and ReAct for slot finding."""
    try:
        # 1. Extract structured meeting information using multi-stage extraction
        meeting_info = extract_structured_meeting_info(question)
        if "Error" in meeting_info:
            return "Error extracting meeting information."

        # 2. Use ReAct pattern to find a valid meeting slot
        best_slot = find_meeting_slot_with_react(meeting_info)
        if "Error" in best_slot:
            return "Error finding a valid meeting slot."

        return best_slot
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def extract_structured_meeting_info(question):
    """Extracts meeting information in a structured format using LLM with embedded examples."""
    system_instruction = "You are an expert at extracting meeting scheduling details into a structured JSON format."
    prompt = f"""
    Extract structured meeting information from the following text. Return the information as a JSON object.

    Example:
    Input: You need to schedule a meeting for John and Jane for 30 minutes between 9:00 and 17:00 on Monday. John is busy 10:00-11:00, Jane is busy 13:00-14:00.
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

    Input: You need to schedule a meeting for John, Jane, and Peter for one hour on Tuesday. John is free all day. Jane is busy 11:00-12:00. Peter is busy 14:00-15:00.
    Output:
    {{
      "participants": ["John", "Jane", "Peter"],
      "duration": 60,
      "days": ["Tuesday"],
      "work_hours": ["9:00", "17:00"],
      "schedules": {{
        "John": [],
        "Jane": [["11:00", "12:00"]],
        "Peter": [["14:00", "15:00"]]
      }}
    }}
    
    Input: {question}
    Output:
    """
    try:
        extracted_info = call_llm(prompt, system_instruction)
        # Attempt to parse the JSON. If it fails, re-prompt for corrected JSON.
        try:
            meeting_info = json.loads(extracted_info)
            return meeting_info
        except json.JSONDecodeError as e:
            # Reprompt LLM with error feedback
            correction_prompt = f"""
            The previous JSON was invalid and could not be parsed due to the following error: {str(e)}.
            Please provide a corrected JSON object with the same structure, taking into account the original input: {question}.
            """
            corrected_info = call_llm(correction_prompt, system_instruction)
            try:
                meeting_info = json.loads(corrected_info)
                return meeting_info
            except json.JSONDecodeError:
                return "Error: Could not extract valid JSON after multiple attempts." # If it fails the second time then error.
    except Exception as e:
        return f"Error extracting info: {str(e)}"

def find_meeting_slot_with_react(meeting_info):
    """Finds a valid meeting slot using the ReAct pattern with LLM reasoning."""
    system_instruction = "You are a ReAct agent for finding valid meeting times, alternating between reasoning and actions."
    prompt = f"""
    You are provided with meeting information in JSON format. Use the ReAct pattern to find a valid meeting slot.
    
    Meeting Information:
    {json.dumps(meeting_info)}
    
    Here's how to use the ReAct pattern:
    1. REASON: Start by carefully reviewing the meeting information and identify available time slots.
    2. ACTION: Propose a potential meeting slot (e.g., "Monday, 14:00 - 14:30").
    3. OBSERVATION: Check if the proposed slot conflicts with any participant's schedule.
    4. Repeat steps 1-3 until a valid slot is found, or you determine no valid slot exists.
    5. FINISH: Once a valid slot is found, output the result (e.g., "Valid meeting slot: Monday, 14:00 - 14:30"). If no valid slot can be found, output "No valid meeting slot found".
    
    Example:
    Meeting Information:
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
    
    Thought 1: Okay, let's find a 30-minute slot on Monday between 9:00 and 17:00 that works for both John and Jane, considering their schedules.
    Action 1: Propose Monday, 9:00 - 9:30
    Observation 1: John is available. Jane is available. The proposed slot is valid.
    Action 2: FINISH Valid meeting slot: Monday, 9:00 - 9:30

    Example 2:
    Meeting Information:
    {{
      "participants": ["Alice", "Bob"],
      "duration": 60,
      "days": ["Tuesday"],
      "work_hours": ["10:00", "16:00"],
      "schedules": {{
        "Alice": [["11:00", "12:00"]],
        "Bob": [["14:00", "15:00"]]
      }}
    }}

    Thought 1: Let's find a 60-minute slot on Tuesday between 10:00 and 16:00 that works for Alice and Bob, considering their schedules.
    Action 1: Propose Tuesday, 10:00 - 11:00
    Observation 1: Alice is available. Bob is available. The proposed slot is valid.
    Action 2: FINISH Valid meeting slot: Tuesday, 10:00 - 11:00
    
    Let's begin! Start with Thought 1.
    """
    try:
        react_response = call_llm(prompt, system_instruction)
        # Extract result. If finding result failed it should say No valid meeting slot found
        if "Valid meeting slot" in react_response:
            return react_response.split("Valid meeting slot: ")[1].strip()
        else:
            return "No valid meeting slot found"

    except Exception as e:
        return f"Error finding slot: {str(e)}"