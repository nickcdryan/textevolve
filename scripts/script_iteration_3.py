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

def extract_constraints_and_generate_options(question):
    """Extracts constraints and generates potential time slots using LLM. Includes examples."""
    system_instruction = "You are an expert meeting scheduler. Extract constraints AND generate potential time options. Return JSON."
    prompt = f"""
    Given the meeting scheduling question, extract all constraints AND generate 3 potential meeting time options.
    Present the output as a JSON object. Include extracted participants' schedules as well.

    Example:
    Question: You need to schedule a meeting for John and Mary for half an hour between 9:00 to 17:00 on Monday. John is busy from 10:00-11:00, Mary is busy from 14:00-15:00.
    Output:
    {{
      "participants": ["John", "Mary"],
      "duration": "half an hour",
      "day": "Monday",
      "start_time": "9:00",
      "end_time": "17:00",
      "John_schedule": ["10:00-11:00"],
      "Mary_schedule": ["14:00-15:00"],
      "potential_times": ["9:00-9:30", "11:00-11:30", "16:00-16:30"]
    }}

    Question: {question}
    Output:
    """
    return call_llm(prompt, system_instruction)

def filter_and_verify_options(question, extracted_data_json):
    """Filters options, verifies constraints, and provides feedback. Includes examples."""
    system_instruction = "You are an expert at verifying meeting times. Filter options and provide validation feedback."
    prompt = f"""
    Given the question and extracted data, filter the potential meeting times to return the best SINGLE valid option.
    Consider participant schedules and duration. If no valid time exists, respond with 'No valid time'.

    Example:
    Question: You need to schedule a meeting for John and Mary for half an hour between 9:00 to 17:00 on Monday. John is busy from 10:00-11:00, Mary is busy from 14:00-15:00.
    Extracted Data:
    {{
      "participants": ["John", "Mary"],
      "duration": "half an hour",
      "day": "Monday",
      "start_time": "9:00",
      "end_time": "17:00",
      "John_schedule": ["10:00-11:00"],
      "Mary_schedule": ["14:00-15:00"],
      "potential_times": ["9:00-9:30", "11:00-11:30", "16:00-16:30"]
    }}
    Valid Time: 9:00-9:30

    Question: {question}
    Extracted Data: {extracted_data_json}
    Valid Time:
    """
    return call_llm(prompt, system_instruction)

def verify_extracted_data(question, extracted_data_json):
    """Verifies the extracted data for correctness and completeness."""
    system_instruction = "You are a meticulous data verifier. Check for accuracy and completeness."
    prompt = f"""
    You are an expert at verifying that extracted data matches the original question.

    Example:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday between 9am and 5pm. John is busy 10-11am, Mary is busy 2-3pm.
    Extracted Data:
    {{
      "participants": ["John", "Mary"],
      "duration": "30 minutes",
      "day": "Monday",
      "start_time": "9am",
      "end_time": "5pm",
      "John_schedule": ["10-11am"],
      "Mary_schedule": ["2-3pm"]
    }}
    Verification: Data is accurate and complete.

    Question: {question}
    Extracted Data: {extracted_data_json}
    Verification:
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule meetings with data extraction verification."""
    try:
        # Extract constraints and generate options
        extracted_data_json = extract_constraints_and_generate_options(question)

        # Verify Extracted Data
        verification_result = verify_extracted_data(question, extracted_data_json)

        if "accurate and complete" not in verification_result.lower():
            return "Error: Inaccurate data extraction."

        # Filter and verify options
        valid_time = filter_and_verify_options(question, extracted_data_json)

        if "No valid time" not in valid_time:
            return "Here is the proposed time: Monday, " + valid_time
        else:
            return "Could not find a valid meeting time."

    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error processing the request."