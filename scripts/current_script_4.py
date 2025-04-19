import os
import re
import json
import math

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

def extract_constraints_with_validation(question, max_attempts=3):
    """Extracts constraints and validates them using a loop."""
    system_instruction = "You are an expert at extracting and validating meeting constraints."
    prompt = f"""
    Extract meeting constraints from the question and VALIDATE the extracted constraints.

    Example:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday between 9am and 5pm. John is busy 10-11am, Mary is busy 2-3pm.
    Output:
    {{
      "participants": ["John", "Mary"],
      "duration": "30 minutes",
      "day": "Monday",
      "start_time": "9am",
      "end_time": "5pm",
      "John_schedule": ["10-11am"],
      "Mary_schedule": ["2-3pm"],
      "is_valid": true,
      "validation_feedback": "All constraints are valid."
    }}

    Question: {question}
    Output:
    """

    for attempt in range(max_attempts):
        extracted_data = call_llm(prompt, system_instruction)
        try:
            data = json.loads(extracted_data)
            # Validation step: Ask LLM to verify its own extracted data
            verification_prompt = f"""
            You extracted the following data:
            {data}
            From the question:
            {question}
            Is the extracted data valid (Yes/No)? Explain any issues.
            """
            verification_result = call_llm(verification_prompt, system_instruction)
            if "Yes" in verification_result:
                data["is_valid"] = True
                data["validation_feedback"] = "Data is valid."
                return data
            else:
                data["is_valid"] = False
                data["validation_feedback"] = verification_result
                #Refine data with feedback, but only if attempts are remaining

        except Exception as e:
            print(f"Error extracting/validating constraints: {e}")
            return {"is_valid": False, "validation_feedback": f"Error during processing: {e}"}

    return {"is_valid": False, "validation_feedback": "Could not extract valid constraints after multiple attempts."}

def generate_time_slots(constraints, max_slots=5):
    """Generates a list of potential time slots using LLM."""
    system_instruction = "You are an expert at generating potential meeting time slots based on given constraints."
    prompt = f"""
    Given the following constraints, generate a list of {max_slots} potential time slots. Provide times in format HH:MM-HH:MM.

    Example:
    Constraints:
    {{
      "participants": ["John", "Mary"],
      "duration": "30 minutes",
      "day": "Monday",
      "start_time": "9am",
      "end_time": "5pm",
      "John_schedule": ["10:00-11:00"],
      "Mary_schedule": ["2:00-3:00"],
      "is_valid": true
    }}
    Potential Time Slots: ["09:00-09:30", "09:30-10:00", "11:00-11:30", "11:30-12:00", "12:00-12:30"]

    Constraints: {constraints}
    Potential Time Slots:
    """
    return call_llm(prompt, system_instruction)

def filter_time_slots(question, constraints, time_slots_str):
    """Filters the generated time slots based on given constraints using LLM and provides one time."""
    system_instruction = "You are an expert at filtering time slots and selecting ONE valid option."
    prompt = f"""
    Given the question, constraints, and potential time slots, filter the slots to suggest ONE valid meeting time.

    Example:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday between 9am and 5pm. John is busy 10-11am, Mary is busy 2-3pm.
    Constraints:
    {{
      "participants": ["John", "Mary"],
      "duration": "30 minutes",
      "day": "Monday",
      "start_time": "9am",
      "end_time": "5pm",
      "John_schedule": ["10:00-11:00"],
      "Mary_schedule": ["2:00-3:00"],
      "is_valid": true
    }}
    Time Slots: ["09:00-09:30", "09:30-10:00", "11:00-11:30", "11:30-12:00", "12:00-12:30"]
    Valid Meeting Time: Monday, 09:00-09:30

    Question: {question}
    Constraints: {constraints}
    Time Slots: {time_slots_str}
    Valid Meeting Time:
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to orchestrate meeting scheduling."""
    try:
        # 1. Extract constraints with validation
        constraints_data = extract_constraints_with_validation(question)
        if not constraints_data["is_valid"]:
            return f"Error: Could not extract valid constraints. {constraints_data['validation_feedback']}"

        # 2. Generate potential time slots
        time_slots_str = generate_time_slots(json.dumps(constraints_data))

        # 3. Filter and suggest ONE valid time
        valid_meeting_time = filter_time_slots(question, json.dumps(constraints_data), time_slots_str)

        return f"Here is the proposed time: {valid_meeting_time}"

    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error processing the request."