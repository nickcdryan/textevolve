import os
import json
import re

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

def extract_meeting_constraints(question):
    """Extract meeting constraints from the input question using LLM with examples."""
    system_instruction = "You are an expert at extracting meeting constraints from text."
    prompt = f"""
    Extract the following constraints from the question: participants, duration, days, start_time, end_time, existing schedules, preferences.

    Example:
    Question: You need to schedule a meeting for Carol and Mark for half an hour between the work hours of 9:00 to 17:00 on Monday. Carol has blocked their calendar on Monday during 10:00 to 11:00; Mark has blocked their calendar on Monday during 9:30 to 10:00.
    Extracted Constraints: {{"participants": ["Carol", "Mark"], "duration": "half an hour", "days": ["Monday"], "start_time": "9:00", "end_time": "17:00", "existing_schedules": {{"Carol": {{"Monday": ["10:00-11:00"]}}, "Mark": {{"Monday": ["9:30-10:00"]}}}}, "preferences": {{}}}}

    Example 2:
    Question: You need to schedule a meeting for John, Jane, and Peter for 1 hour on Tuesday or Wednesday between 10:00 and 16:00. John is busy on Tuesday from 11:00-13:00, Jane is busy on Wednesday from 14:00-15:00, and Peter has no conflicts.
    Extracted Constraints: {{"participants": ["John", "Jane", "Peter"], "duration": "1 hour", "days": ["Tuesday", "Wednesday"], "start_time": "10:00", "end_time": "16:00", "existing_schedules": {{"John": {{"Tuesday": ["11:00-13:00"]}}, "Jane": {{"Wednesday": ["14:00-15:00"]}}, "Peter": {{}}}}, "preferences": {{}}}}

    Example 3:
    Question: You need to schedule a meeting for Scott, Nicholas, Donna, Vincent and Ann for half an hour between the work hours of 9:00 to 17:00 on Monday. Scott is free the entire day. Nicholas has blocked their calendar on Monday during 11:00 to 11:30, 15:30 to 16:00; Donna has meetings on Monday during 9:30 to 10:00, 12:00 to 12:30, 14:00 to 14:30, 16:00 to 16:30; Vincent is busy on Monday during 9:30 to 11:00, 11:30 to 12:00, 13:30 to 14:30, 15:30 to 16:30; Ann has meetings on Monday during 9:30 to 11:00, 12:00 to 13:00, 14:00 to 15:00, 16:30 to 17:00; Vincent do not want to meet on Monday after 14:30.
    Extracted Constraints: {{"participants": ["Scott", "Nicholas", "Donna", "Vincent", "Ann"], "duration": "half an hour", "days": ["Monday"], "start_time": "9:00", "end_time": "17:00", "existing_schedules": {{"Scott": {{"Monday": []}}, "Nicholas": {{"Monday": ["11:00-11:30", "15:30-16:00"]}}, "Donna": {{"Monday": ["9:30-10:00", "12:00-12:30", "14:00-14:30", "16:00-16:30"]}}, "Vincent": {{"Monday": ["9:30-11:00", "11:30-12:00", "13:30-14:30", "15:30-16:30"]}}, "Ann": {{"Monday": ["9:30-11:00", "12:00-13:00", "14:00-15:00", "16:30-17:00"]}}}}, "preferences": {{"Vincent": {{"Monday": "before 14:30"}}}}}}

    Question: {question}
    """
    return call_llm(prompt, system_instruction)

def propose_meeting_time(constraints_json):
    """Propose a meeting time using LLM reasoning with examples."""
    system_instruction = "You are an expert at proposing meeting times given constraints."
    prompt = f"""
    Given these meeting constraints, propose a meeting time that satisfies all participants' schedules.

    Example:
    Constraints: {{"participants": ["Carol", "Mark"], "duration": "half an hour", "days": ["Monday"], "start_time": "9:00", "end_time": "17:00", "existing_schedules": {{"Carol": {{"Monday": ["10:00-11:00"]}}, "Mark": {{"Monday": ["9:30-10:00"]}}}}, "preferences": {{}}}}
    Reasoning: Carol is busy from 10:00-11:00 and Mark is busy from 9:30-10:00. A time that works for both is 9:00-9:30.
    Proposed Time: Here is the proposed time: Monday, 9:00 - 9:30

    Example 2:
    Constraints: {{"participants": ["John", "Jane", "Peter"], "duration": "1 hour", "days": ["Tuesday", "Wednesday"], "start_time": "10:00", "end_time": "16:00", "existing_schedules": {{"John": {{"Tuesday": ["11:00-13:00"]}}, "Jane": {{"Wednesday": ["14:00-15:00"]}}, "Peter": {{}}}}, "preferences": {{}}}}
    Reasoning: John is busy on Tuesday from 11:00-13:00, Jane is busy on Wednesday from 14:00-15:00, and Peter is free both days. A time that works for all is Tuesday from 10:00-11:00.
    Proposed Time: Here is the proposed time: Tuesday, 10:00 - 11:00

    Example 3:
    Constraints: {{"participants": ["Scott", "Nicholas", "Donna", "Vincent", "Ann"], "duration": "half an hour", "days": ["Monday"], "start_time": "9:00", "end_time": "17:00", "existing_schedules": {{"Scott": {{"Monday": []}}, "Nicholas": {{"Monday": ["11:00-11:30", "15:30-16:00"]}}, "Donna": {{"Monday": ["9:30-10:00", "12:00-12:30", "14:00-14:30", "16:00-16:30"]}}, "Vincent": {{"Monday": ["9:30-11:00", "11:30-12:00", "13:30-14:30", "15:30-16:30"]}}, "Ann": {{"Monday": ["9:30-11:00", "12:00-13:00", "14:00-15:00", "16:30-17:00"]}}}}, "preferences": {{"Vincent": {{"Monday": "before 14:30"}}}}}}
    Reasoning: Scott is available all day. Nicholas is busy from 11:00-11:30 and 15:30-16:00. Donna is busy from 9:30-10:00, 12:00-12:30, 14:00-14:30 and 16:00-16:30. Vincent is busy from 9:30-11:00, 11:30-12:00, 13:30-14:30, and 15:30-16:30 and wants to meet before 14:30. Ann is busy from 9:30-11:00, 12:00-13:00, 14:00-15:00 and 16:30-17:00. A time that works for everyone is 13:00-13:30.
    Proposed Time: Here is the proposed time: Monday, 13:00 - 13:30

    Constraints: {constraints_json}
    Reasoning: Let's analyze the constraints and determine the best possible meeting time
    """
    return call_llm(prompt, system_instruction)

def verify_solution(question, proposed_time):
    """Verify if the proposed solution satisfies all requirements using LLM with examples."""
    system_instruction = "You are a critical evaluator who verifies meeting schedules."
    prompt = f"""
    Verify if the proposed meeting time satisfies all requirements in the question and that no constraints are violated.

    Example:
    Question: You need to schedule a meeting for Carol and Mark for half an hour between 9:00 to 17:00 on Monday. Carol has blocked their calendar on Monday during 10:00 to 11:00; Mark has blocked their calendar on Monday during 9:30 to 10:00.
    Proposed Time: Monday, 9:00 - 9:30
    Verification: The proposed time satisfies all requirements because it is on Monday, between 9:00 and 17:00, and does not conflict with Carol's or Mark's schedules.
    Result: Valid

    Example 2:
    Question: You need to schedule a meeting for John, Jane, and Peter for 1 hour on Tuesday or Wednesday between 10:00 and 16:00. John is busy on Tuesday from 11:00-13:00, Jane is busy on Wednesday from 14:00-15:00, and Peter has no conflicts.
    Proposed Time: Tuesday, 11:30 - 12:30
    Verification: The proposed time violates John's schedule on Tuesday from 11:00-13:00.
    Result: Invalid

    Example 3:
    Question: You need to schedule a meeting for Scott, Nicholas, Donna, Vincent and Ann for half an hour between the work hours of 9:00 to 17:00 on Monday. Scott is free the entire day. Nicholas has blocked their calendar on Monday during 11:00 to 11:30, 15:30 to 16:00; Donna has meetings on Monday during 9:30 to 10:00, 12:00 to 12:30, 14:00 to 14:30, 16:00 to 16:30; Vincent is busy on Monday during 9:30 to 11:00, 11:30 to 12:00, 13:30 to 14:30, 15:30 to 16:30; Ann has meetings on Monday during 9:30 to 11:00, 12:00 to 13:00, 14:00 to 15:00, 16:30 to 17:00; Vincent do not want to meet on Monday after 14:30.
    Proposed Time: Monday, 13:00 - 13:30
    Verification: Scott is available. Nicholas is available. Donna is available. Vincent is available and the time satisfies their preference of before 14:30. Ann is available. Thus, all constraints are met.
    Result: Valid

    Question: {question}
    Proposed Time: {proposed_time}
    Verification: Let's meticulously check if the proposed meeting time is valid.
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule a meeting."""
    try:
        # Extract meeting constraints
        constraints_json = extract_meeting_constraints(question)

        # Propose a meeting time
        proposed_time = propose_meeting_time(constraints_json)

        # Verify the solution
        verification_result = verify_solution(question, proposed_time)

        if "Invalid" in verification_result:
            return "No valid meeting time found."

        return proposed_time
    except Exception as e:
        return f"Error: {str(e)}"