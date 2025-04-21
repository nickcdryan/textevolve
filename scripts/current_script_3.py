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

def main(question):
    """Main function to schedule meetings based on constraints."""

    # Step 1: Extract information with a structured extraction prompt, including reasoning steps
    def extract_meeting_details(question_text):
        """Extract meeting details with example-driven structured extraction."""
        system_instruction = "You are an expert at extracting structured meeting data."
        prompt = f"""
        Extract meeting details from the text, following the example below:

        Example:
        Text: You need to schedule a meeting for John and Mary for half an hour between 9:00 and 17:00 on Monday. John is busy Monday 10:00-11:00. Mary is busy Monday 14:00-15:00.
        Reasoning:
        1. Participants: Identify the people involved: John and Mary.
        2. Duration: Note the meeting duration: half an hour.
        3. Days: Determine the possible days: Monday.
        4. Work Hours: Capture the work hours: 9:00 to 17:00.
        5. Schedules: Extract each participant's schedule: John is busy 10:00-11:00, Mary is busy 14:00-15:00 on Monday.

        Extracted Details:
        {{
          "participants": ["John", "Mary"],
          "duration": "30 minutes",
          "days": ["Monday"],
          "work_hours": "9:00 to 17:00",
          "John_schedule": "Monday: 10:00-11:00",
          "Mary_schedule": "Monday: 14:00-15:00"
        }}

        Now, extract details from this text:
        {question_text}
        """
        return call_llm(prompt, system_instruction)

    extracted_details = extract_meeting_details(question)
    print(f"Extracted Details: {extracted_details}")

    # Step 2: Validate extracted details for completeness and correctness
    def validate_extraction(extracted_data, original_question):
        """Validates the extracted data with a verification prompt."""
        system_instruction = "You are an expert at validating extracted meeting data."
        prompt = f"""
        Validate the extracted meeting data below, checking for completeness and correctness against the original question.

        Example:
        Question: Schedule a meeting for Alice and Bob on Tuesday. Alice is busy 9am-10am.
        Extracted: {{"participants": ["Alice"], "days": ["Tuesday"]}}
        Validation: INCOMPLETE. Missing participants and schedule information.

        Question: Schedule a meeting for Charlie and David on Wednesday. Charlie is busy 2pm-3pm. David is busy 4pm-5pm.
        Extracted: {{"participants": ["Charlie", "David"], "days": ["Wednesday"], "Charlie_schedule": "2pm-3pm"}}
        Validation: INCOMPLETE. Missing David's schedule information.

        Question: {original_question}
        Extracted: {extracted_data}
        Validation:
        """
        return call_llm(prompt, system_instruction)

    validation_result = validate_extraction(extracted_details, question)
    print(f"Validation Result: {validation_result}")

    # Step 3: Refine extraction based on validation feedback (if needed)
    if "INCOMPLETE" in validation_result:
        def refine_extraction(extracted_data, validation_result, original_question):
            """Refines extraction based on feedback using an LLM call."""
            system_instruction = "You are an expert at refining meeting data extractions."
            prompt = f"""
            Refine the extracted meeting data to address the issues identified in the validation.

            Example:
            Original Question: Schedule a meeting for Eric and Fiona. Eric is busy 11am-12pm.
            Extracted: {{"participants": ["Eric"], "days": []}}
            Validation: INCOMPLETE. Missing participants, schedule information, and day details.
            Refined Extraction: {{"participants": ["Eric", "Fiona"], "days": [], "Eric_schedule": "11am-12pm"}}

            Original Question: {original_question}
            Extracted: {extracted_data}
            Validation: {validation_result}
            Refined Extraction:
            """
            return call_llm(prompt, system_instruction)

        refined_details = refine_extraction(extracted_details, validation_result, question)
        print(f"Refined Details: {refined_details}")
        extracted_details = refined_details # Update the data with refined details

    # Step 4: Identify available time slots (simplified for brevity - the key is the initial extraction)
    def find_available_slots(data):
        """Identifies available time slots based on extracted data (placeholder)."""
        system_instruction = "You are an expert at scheduling meetings given extracted data."
        prompt = f"""Given this extracted data, find ONE suitable 30 minute time: {data}."""
        return call_llm(prompt, system_instruction)

    available_slots = find_available_slots(extracted_details)
    print(f"Available Slots: {available_slots}")
    return "Here is the proposed time: " + available_slots