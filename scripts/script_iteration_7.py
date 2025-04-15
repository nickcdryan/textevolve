import os
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

def extract_meeting_constraints(text):
    """Extract meeting constraints using an LLM with embedded examples."""
    system_instruction = "You are an expert meeting scheduler. Extract meeting constraints from the given text."

    prompt = f"""
    You will be given a text describing a meeting scheduling scenario. Your task is to extract all relevant constraints.

    Example:
    Input:
    You need to schedule a meeting for Daniel and Kathleen for half an hour between 9:00 to 17:00 on Monday. Daniel has no meetings. Kathleen is busy 14:30 to 15:30.
    
    Reasoning:
    1. Participants: Daniel, Kathleen
    2. Duration: 30 minutes
    3. Days: Monday
    4. Schedules: Daniel-Free, Kathleen-Busy 14:30-15:30

    Output:
    {{
        "participants": ["Daniel", "Kathleen"],
        "duration": 30,
        "days": ["Monday"],
        "schedules": {{
            "Daniel": [["Monday", "9:00", "17:00", "free"]],
            "Kathleen": [["Monday", "14:30", "15:30", "busy"]]
        }},
        "preferences": []
    }}

    Now, extract the meeting constraints from the following text:
    {text}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        print(f"Error in constraint extraction: {e}")
        return None


def find_available_time_slots(constraints_json):
    """Find available time slots based on extracted constraints.  LLM Driven."""
    system_instruction = "You are a meeting scheduling expert. Find available time slots based on the provided constraints, considering earliest availability."

    prompt = f"""
    You are provided with a JSON object that contains meeting constraints. Find the *earliest* suitable time slot.

    Example:
    Input:
    {{
        "participants": ["Daniel", "Kathleen"],
        "duration": 30,
        "days": ["Monday"],
        "schedules": {{
            "Daniel": [["Monday", "9:00", "17:00", "free"]],
            "Kathleen": [["Monday", "14:30", "15:30", "busy"]]
        }},
        "preferences": []
    }}
    Reasoning:
    1. Daniel is free all day.
    2. Kathleen is busy 14:30-15:30.
    3. Earliest time must be before 14:30.
    4. Suggest earliest valid time Monday 9:00-9:30.

    Output:
    Here is the proposed time: Monday, 9:00 - 9:30

    Now, find the *earliest* suitable time slot based on these constraints.
    Constraints:
    {constraints_json}
    """

    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        print(f"Error finding time slots: {e}")
        return None


def verify_solution(question, proposed_solution):
    """Verify if the proposed solution is valid using an LLM."""
    system_instruction = "You are an expert solution checker. Verify the proposed solution against all constraints."

    prompt = f"""
    You are given a question and a proposed solution. Verify if the proposed solution is valid.

    Example:
    Question:
    Schedule Daniel and Kathleen for 30 minutes on Monday between 9:00-17:00. Daniel is free. Kathleen is busy 14:30-15:30.
    Proposed solution:
    Here is the proposed time: Monday, 13:30 - 14:00

    Reasoning:
    1. Daniel is available at 13:30-14:00.
    2. Kathleen is not busy at 13:30-14:00.
    3. The time is between 9:00 and 17:00.
    4. All constraints satisfied.

    Output:
    VALID: The proposed solution satisfies all constraints.

    Now, verify the proposed solution for the following new question.
    Question:
    {question}
    Proposed solution:
    {proposed_solution}
    """

    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        print(f"Error during solution verification: {e}")
        return None


def main(question):
    """Main function to solve the meeting scheduling problem."""
    # Step 1: Extract meeting constraints
    constraints_json = extract_meeting_constraints(question)
    if not constraints_json:
        return "Could not extract meeting constraints."

    # Step 2: Find available time slots
    proposed_solution = find_available_time_slots(constraints_json)
    if not proposed_solution:
        return "Could not find available time slots."

    # Step 3: Verify the solution
    verification_result = verify_solution(question, proposed_solution)
    if not verification_result:
        return "Could not verify the proposed solution."

    # Step 4: Return the result
    return proposed_solution if "VALID" in verification_result else "No valid solution found."