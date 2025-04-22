import os
import re
import math
import json

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
    """
    This script approaches the scheduling problem by first decomposing it into sub-problems using the LLM,
    then solving each sub-problem with specialized functions and finally, combining the solutions.
    This is a new approach compared to the previous attempts which mostly use the solve_with_react_pattern.

    Hypothesis: Decomposing the problem into sub-problems will provide better results than the prior approach.
    """
    try:
        # Decompose the problem into sub-problems using LLM
        decomposition = decompose_problem(question)

        # Extract information from the question
        information = extract_information(question)

        # Identify constraints from the question
        constraints = identify_constraints(question)

        # Find available time slots based on information and constraints
        available_time_slots = find_available_time_slots(information, constraints)

        # Format the final answer
        final_answer = format_answer(available_time_slots)

        return final_answer

    except Exception as e:
        return f"Error: {str(e)}"

def decompose_problem(question):
    """Decompose the problem into sub-problems using LLM."""
    system_instruction = "You are an expert at decomposing complex problems into simpler sub-problems."
    prompt = f"""
    Decompose the following problem into a list of sub-problems that need to be solved in order to find the final solution.

    Example:
    Problem: Schedule a meeting for John and Mary for 30 minutes on Monday between 9am and 5pm. John is busy from 10am to 11am and Mary is busy from 2pm to 3pm.
    Sub-problems:
    1. Identify the participants: John and Mary.
    2. Identify the duration of the meeting: 30 minutes.
    3. Identify the day of the meeting: Monday.
    4. Identify the work hours: 9am to 5pm.
    5. Identify John's busy time: 10am to 11am.
    6. Identify Mary's busy time: 2pm to 3pm.
    7. Find a time slot that works for both John and Mary.

    Problem: {question}
    Sub-problems:
    """
    return call_llm(prompt, system_instruction)

def extract_information(question):
    """Extract information from the question using LLM."""
    system_instruction = "You are an expert at extracting information from text."
    prompt = f"""
    Extract the following information from the text:
    - Participants
    - Duration of the meeting
    - Day of the meeting
    - Work hours
    - Existing schedules for everyone during the day
    - Any preferences on the meeting time

    Example:
    Text: You need to schedule a meeting for Joyce, Christine and Alexander for half an hour between the work hours of 9:00 to 17:00 on Monday. Joyce has meetings on Monday during 11:00 to 11:30, 13:30 to 14:00, 14:30 to 16:30; Christinehas no meetings the whole day. Alexander has meetings on Monday during 9:00 to 11:00, 12:00 to 12:30, 13:30 to 15:00, 15:30 to 16:00, 16:30 to 17:00; Christine can not meet on Monday before 12:00.
    Information:
    {{
        "participants": ["Joyce", "Christine", "Alexander"],
        "duration": "30 minutes",
        "day": "Monday",
        "work_hours": "9:00 to 17:00",
        "Joyce": ["11:00 to 11:30", "13:30 to 14:00", "14:30 to 16:30"],
        "Christine": [],
        "Alexander": ["9:00 to 11:00", "12:00 to 12:30", "13:30 to 15:00", "15:30 to 16:00", "16:30 to 17:00"],
        "Christine_preference": "not before 12:00"
    }}

    Text: {question}
    Information:
    """
    return call_llm(prompt, system_instruction)

def identify_constraints(question):
    """Identify constraints from the question using LLM."""
    system_instruction = "You are an expert at identifying constraints from text."
    prompt = f"""
    Identify all the constraints from the text.

    Example:
    Text: You need to schedule a meeting for Joyce, Christine and Alexander for half an hour between the work hours of 9:00 to 17:00 on Monday. Joyce has meetings on Monday during 11:00 to 11:30, 13:30 to 14:00, 14:30 to 16:30; Christinehas no meetings the whole day. Alexander has meetings on Monday during 9:00 to 11:00, 12:00 to 12:30, 13:30 to 15:00, 15:30 to 16:00, 16:30 to 17:00; Christine can not meet on Monday before 12:00.
    Constraints:
    - Meeting duration: 30 minutes
    - Day: Monday
    - Work hours: 9:00 to 17:00
    - Joyce is busy from 11:00 to 11:30, 13:30 to 14:00, 14:30 to 16:30
    - Alexander is busy from 9:00 to 11:00, 12:00 to 12:30, 13:30 to 15:00, 15:30 to 16:00, 16:30 to 17:00
    - Christine can not meet before 12:00

    Text: {question}
    Constraints:
    """
    return call_llm(prompt, system_instruction)

def find_available_time_slots(information, constraints):
    """Find available time slots based on information and constraints."""
    system_instruction = "You are an expert at finding available time slots."
    prompt = f"""
    Find available time slots based on the following information and constraints.

    Information:
    {information}

    Constraints:
    {constraints}

    Example:
    Information:
    {{
        "participants": ["Joyce", "Christine", "Alexander"],
        "duration": "30 minutes",
        "day": "Monday",
        "work_hours": "9:00 to 17:00",
        "Joyce": ["11:00 to 11:30", "13:30 to 14:00", "14:30 to 16:30"],
        "Christine": [],
        "Alexander": ["9:00 to 11:00", "12:00 to 12:30", "13:30 to 15:00", "15:30 to 16:00", "16:30 to 17:00"],
        "Christine_preference": "not before 12:00"
    }}
    Constraints:
    - Meeting duration: 30 minutes
    - Day: Monday
    - Work hours: 9:00 to 17:00
    - Joyce is busy from 11:00 to 11:30, 13:30 to 14:00, 14:30 to 16:30
    - Alexander is busy from 9:00 to 11:00, 12:00 to 12:30, 13:30 to 15:00, 15:30 to 16:00, 16:30 to 17:00
    - Christine can not meet before 12:00
    Available time slots:
    Monday, 12:30 - 13:00

    Information:
    {information}
    Constraints:
    {constraints}
    Available time slots:
    """
    return call_llm(prompt, system_instruction)

def format_answer(available_time_slots):
    """Format the final answer."""
    system_instruction = "You are an expert at formatting answers."
    prompt = f"""
    Format the following available time slots into a final answer.

    Example:
    Available time slots:
    Monday, 12:30 - 13:00
    Final answer:
    Here is the proposed time: Monday, 12:30 - 13:00

    Available time slots:
    {available_time_slots}
    Final answer:
    """
    return call_llm(prompt, system_instruction)