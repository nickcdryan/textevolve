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

def extract_data_with_verifier(text):
    """Extract scheduling constraints and verifies if extraction is correct using multi-agent system."""
    system_instruction_extraction = "You are an expert information extraction specialist."
    prompt_extraction = f"""
    Extract the scheduling constraints from the text below.

    Example Input:
    You need to schedule a meeting for Carol and Mark for half an hour between the work hours of 9:00 to 17:00 on Monday.
    Carol has blocked their calendar on Monday during 10:00 to 11:00, 14:30 to 15:00, 15:30 to 17:00;
    Mark has blocked their calendar on Monday during 9:30 to 10:00, 10:30 to 17:00;

    Extracted data:
    {{
        "participants": ["Carol", "Mark"],
        "duration": "30 minutes",
        "day": "Monday",
        "time_range_start": "9:00",
        "time_range_end": "17:00",
        "conflicts": {{
            "Carol": ["10:00-11:00", "14:30-15:00", "15:30-17:00"],
            "Mark": ["9:30-10:00", "10:30-17:00"]
        }}
    }}

    Now, extract the data from this new text:
    {text}
    """
    extracted_data = call_llm(prompt_extraction, system_instruction_extraction)

    system_instruction_verification = "You are an expert verifier, and you check data extractions for correctness and consistency."
    prompt_verification = f"""
    You are given the following extracted data and the original text. Verify if the extracted data is correct and consistent with the original text.
    
    Original Text:
    {text}
    
    Extracted Data:
    {extracted_data}

    Example of correct verification:
    Original Text: You need to schedule a meeting for Carol and Mark for half an hour between the work hours of 9:00 to 17:00 on Monday. Carol has blocked their calendar on Monday during 10:00 to 11:00, 14:30 to 15:00, 15:30 to 17:00; Mark has blocked their calendar on Monday during 9:30 to 10:00, 10:30 to 17:00;
    Extracted Data: {{"participants": ["Carol", "Mark"], "duration": "30 minutes", "day": "Monday", "time_range_start": "9:00", "time_range_end": "17:00", "conflicts": {{"Carol": ["10:00-11:00", "14:30-15:00", "15:30-17:00"], "Mark": ["9:30-10:00", "10:30-17:00"]}}}}
    Verification Result: VALID

    Now, provide the verification result (VALID or INVALID) for the data:
    """
    verification_result = call_llm(prompt_verification, system_instruction_verification)

    if "VALID" in verification_result:
        return extracted_data
    else:
        return "Error: Data extraction is invalid."

def find_available_time(extracted_data):
    """Find available time slot using LLM with explicit reasoning and final verification."""
    system_instruction = "You are an expert at scheduling meetings. Find an available time and verify solution."
    prompt = f"""
    Given the extracted scheduling constraints, find an available meeting time and explicitly verify that solution.
    
    Extracted Data:
    {extracted_data}
    
    Example Input:
    {{"participants": ["Carol", "Mark"], "duration": "30 minutes", "day": "Monday", "time_range_start": "9:00", "time_range_end": "17:00", "conflicts": {{"Carol": ["10:00-11:00", "14:30-15:00", "15:30-17:00"], "Mark": ["9:30-10:00", "10:30-17:00"]}}}}

    Reasoning:
    1. Carol is available from 9:00-10:00, 11:00-14:30, and 15:00-15:30.
    2. Mark is available from 9:00-9:30 and 10:00-10:30.
    3. A 30-minute meeting that works for both is 9:00-9:30.
    4. So, Monday 9:00-9:30 is a solution.
    
    Verification:
    1. Participants: Carol, Mark
    2. Duration: 30 minutes
    3. Available Time Range: 9:00 to 17:00
    4. Day: Monday
    5. Conflicts: Monday 9:00-9:30 does not conflict with the schedule of either.
    6. So, the result is Valid.
    
    Available Time:
    Monday, 9:00-9:30 with verification result Valid.
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to orchestrate scheduling."""
    try:
        extracted_data = extract_data_with_verifier(question)
        if "Error" in extracted_data:
          return extracted_data

        available_time = find_available_time(extracted_data)
        if "Valid" in available_time:
            proposed_time = available_time.split("Available Time:\n")[1].split(" with verification result")[0].strip()
            return "Here is the proposed time: " + proposed_time
        else:
            return "Could not find a valid meeting time."

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return "Error occurred while scheduling."