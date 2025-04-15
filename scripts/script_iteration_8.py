import os
import re
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

def extract_meeting_constraints(text):
    """Extract meeting constraints using an LLM with embedded examples."""
    system_instruction = "You are an expert meeting scheduler. Extract meeting constraints from the given text."

    prompt = f"""
    You will be given a text describing a meeting scheduling scenario. Your task is to extract all relevant constraints in JSON format, including:
    - participants: Names of people involved in the meeting (list of strings).
    - duration: Length of the meeting in minutes (integer).
    - days: Acceptable days for the meeting (list of strings).
    - schedules: Existing schedules of each participant with busy time intervals. Represent each schedule as a list of [day, start_time, end_time] (dictionary).
    - preferences: Any other preferences (list of strings).

    Example:
    Input:
    You need to schedule a meeting for Daniel and Kathleen for half an hour between the work hours of 9:00 to 17:00 on Monday.
    Daniel has no meetings the whole day. Kathleen is busy on Monday during 14:30 to 15:30.

    Reasoning:
    1. Participants: Daniel, Kathleen
    2. Duration: 30 minutes
    3. Days: Monday
    4. Daniel's Schedule: Free all day (9:00-17:00)
    5. Kathleen's Schedule: Busy 14:30-15:30

    Output:
    {{
        "participants": ["Daniel", "Kathleen"],
        "duration": 30,
        "days": ["Monday"],
        "schedules": {{
            "Daniel": [["Monday", "9:00", "17:00"]],
            "Kathleen": [["Monday", "14:30", "15:30"]]
        }},
        "preferences": []
    }}

    Now, extract the meeting constraints from the following text:
    {text}
    """
    try:
        constraints_str = call_llm(prompt, system_instruction)
        constraints = json.loads(constraints_str) # Attempt to parse the JSON
        return constraints
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        return None
    except Exception as e:
        print(f"Error in constraint extraction: {e}")
        return None


def find_available_time_slots(constraints):
    """Find available time slots based on extracted constraints using iterative reasoning."""
    system_instruction = "You are an expert meeting scheduling assistant. Given meeting constraints, find the earliest available time slot."

    prompt = f"""
    You are given a JSON object representing meeting constraints. Your task is to analyze the constraints and determine the *earliest* suitable time slot for the meeting.
    The constraints include participants, duration, days, schedules, and preferences.

    Example:
    Input:
    {{
        "participants": ["Daniel", "Kathleen"],
        "duration": 30,
        "days": ["Monday"],
        "schedules": {{
            "Daniel": [["Monday", "9:00", "17:00"]],
            "Kathleen": [["Monday", "14:30", "15:30"]]
        }},
        "preferences": []
    }}

    Reasoning:
    1. Participants: Daniel, Kathleen
    2. Duration: 30 minutes
    3. Days: Monday
    4. Daniel is available all day (9:00-17:00)
    5. Kathleen is busy from 14:30 to 15:30
    6. Check timeslots chronologically starting from 9:00 on Monday.
    7. Earliest timeslot is Monday 9:00-9:30. Daniel is free. Kathleen is free.
    8. Output earliest possible time.

    Output:
    Here is the proposed time: Monday, 9:00 - 9:30

    Now, using the same chain of thought reasoning process, find the *earliest* suitable time slot based on these new meeting constraints.
    Constraints:
    {json.dumps(constraints)}
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
    You are given a question and a proposed solution. Verify if the proposed solution is valid and satisfies *all* the constraints mentioned in the question.

    Example:
    Question:
    You need to schedule a meeting for Daniel and Kathleen for half an hour between the work hours of 9:00 to 17:00 on Monday.
    Daniel has no meetings the whole day. Kathleen is busy on Monday during 14:30 to 15:30.
    Proposed solution:
    Here is the proposed time: Monday, 13:30 - 14:00

    Reasoning:
    1. Identify participants: Daniel, Kathleen.
    2. Identify proposed time: Monday 13:30-14:00.
    3. Check Daniel's availability at the proposed time: Daniel is available all day.
    4. Check Kathleen's availability at the proposed time: Kathleen is busy 14:30-15:30, so she is available at 13:30-14:00.
    5. Conclusion: Valid because all participants are available and time is within working hours.

    Output:
    VALID: The proposed solution satisfies all constraints.

    Now, using the same chain of thought reasoning process, verify the proposed solution for the following new question.
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
    constraints = extract_meeting_constraints(question)
    if not constraints:
        return "Could not extract meeting constraints."

    # Step 2: Find available time slots
    proposed_solution = find_available_time_slots(constraints)
    if not proposed_solution:
        return "Could not find available time slots."

    # Step 3: Verify the solution
    verification_result = verify_solution(question, proposed_solution)
    if not verification_result:
        return "Could not verify the proposed solution."

    # Step 4: Return the result
    return proposed_solution if "VALID" in verification_result else "No valid solution found."