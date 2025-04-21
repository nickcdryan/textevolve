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

    # Step 1: Extract structured data from the question with embedded examples
    def extract_meeting_data(question_text):
        """Extracts key information from the question using examples."""
        system_instruction = "You are an expert meeting data extractor. Extract key details."
        prompt = f"""
        Extract the following information from the text: participants, duration, days, work hours, existing schedules (for each participant), and any preferences.

        Example 1:
        Text: You need to schedule a meeting for Joyce, Christine and Alexander for half an hour between the work hours of 9:00 to 17:00 on Monday. Joyce has meetings on Monday during 11:00 to 11:30, 13:30 to 14:00, 14:30 to 16:30; Christine has no meetings the whole day. Alexander has meetings on Monday during 9:00 to 11:00, 12:00 to 12:30, 13:30 to 15:00, 15:30 to 16:00, 16:30 to 17:00; Christine can not meet on Monday before 12:00.
        Extracted Data:
        {{
          "participants": ["Joyce", "Christine", "Alexander"],
          "duration": "30 minutes",
          "days": ["Monday"],
          "work_hours": "9:00 to 17:00",
          "Joyce_schedule": "Monday: 11:00-11:30, 13:30-14:00, 14:30-16:30",
          "Christine_schedule": "Monday: None",
          "Alexander_schedule": "Monday: 9:00-11:00, 12:00-12:30, 13:30-15:00, 15:30-16:00, 16:30-17:00",
          "Christine_preference": "Not before 12:00 on Monday"
        }}

        Example 2:
        Text: You need to schedule a meeting for Betty and Scott for half an hour between the work hours of 9:00 to 17:00 on either Monday, Tuesday, Wednesday or Thursday. Betty is busy on Monday during 10:00 to 10:30, 13:30 to 14:00, 15:00 to 15:30, 16:00 to 16:30, Tuesday during 9:00 to 9:30, 11:30 to 12:00, 12:30 to 13:00, 13:30 to 14:00, 16:30 to 17:00, Wednesday during 9:30 to 10:30, 13:00 to 13:30, 14:00 to 14:30, Thursday during 9:30 to 10:00, 11:30 to 12:00, 14:00 to 14:30, 15:00 to 15:30, 16:30 to 17:00; Scott is busy on Monday during 9:30 to 15:00, 15:30 to 16:00, 16:30 to 17:00, Tuesday during 9:00 to 9:30, 10:00 to 11:00, 11:30 to 12:00, 12:30 to 13:30, 14:00 to 15:00, 16:00 to 16:30, Wednesday during 9:30 to 12:30, 13:00 to 13:30, 14:00 to 14:30, 15:00 to 15:30, 16:00 to 16:30, Thursday during 9:00 to 9:30, 10:00 to 10:30, 11:00 to 12:00, 12:30 to 13:00, 15:00 to 16:00, 16:30 to 17:00; Betty can not meet on Monday. Tuesday. Thursday before 15:00. Scott would like to avoid more meetings on Wednesday.
        Extracted Data:
        {{
          "participants": ["Betty", "Scott"],
          "duration": "30 minutes",
          "days": ["Monday", "Tuesday", "Wednesday", "Thursday"],
          "work_hours": "9:00 to 17:00",
          "Betty_schedule": "Monday: 10:00-10:30, 13:30-14:00, 15:00-15:30, 16:00-16:30; Tuesday: 9:00-9:30, 11:30-12:00, 12:30-13:00, 13:30-14:00, 16:30-17:00; Wednesday: 9:30-10:30, 13:00-13:30, 14:00-14:30; Thursday: 9:30-10:00, 11:30-12:00, 14:00-14:30, 15:00-15:30, 16:30-17:00",
          "Scott_schedule": "Monday: 9:30-15:00, 15:30-16:00, 16:30-17:00; Tuesday: 9:00-9:30, 10:00-11:00, 11:30-12:00, 12:30-13:30, 14:00-15:00, 16:00-16:30; Wednesday: 9:30-12:30, 13:00-13:30, 14:00-14:30, 15:00-15:30, 16:00-16:30; Thursday: 9:00-9:30, 10:00-10:30, 11:00-12:00, 12:30-13:00, 15:00-16:00, 16:30-17:00",
          "Betty_preference": "Not before 15:00 on Monday, Tuesday, Thursday",
          "Scott_preference": "Avoid Wednesday"
        }}
        
        Text: {question_text}
        Extracted Data:
        """

        return call_llm(prompt, system_instruction)

    extracted_data = extract_meeting_data(question)
    print(f"Extracted Data: {extracted_data}")

    # Step 2: Identify available time slots based on the extracted data
    def find_available_slots(data):
        """Finds available time slots with embedded examples."""
        system_instruction = "You are an expert at finding available time slots."
        prompt = f"""
        Based on this extracted meeting data, identify available 30-minute time slots that work for all participants, considering work hours and preferences.

        Example:
        Extracted Data:
        {{
          "participants": ["Joyce", "Christine", "Alexander"],
          "duration": "30 minutes",
          "days": ["Monday"],
          "work_hours": "9:00 to 17:00",
          "Joyce_schedule": "Monday: 11:00-11:30, 13:30-14:00, 14:30-16:30",
          "Christine_schedule": "Monday: None",
          "Alexander_schedule": "Monday: 9:00-11:00, 12:00-12:30, 13:30-15:00, 15:30-16:00, 16:30-17:00",
          "Christine_preference": "Not before 12:00 on Monday"
        }}
        Available Slots:
        Monday: 12:30-13:00

        Example:
        Extracted Data:
        {{
          "participants": ["Betty", "Scott"],
          "duration": "30 minutes",
          "days": ["Monday", "Tuesday", "Wednesday", "Thursday"],
          "work_hours": "9:00 to 17:00",
          "Betty_schedule": "Monday: 10:00-10:30, 13:30-14:00, 15:00-15:30, 16:00-16:30; Tuesday: 9:00-9:30, 11:30-12:00, 12:30-13:00, 13:30-14:00, 16:30-17:00; Wednesday: 9:30-10:30, 13:00-13:30, 14:00-14:30; Thursday: 9:30-10:00, 11:30-12:00, 14:00-14:30, 15:00-15:30, 16:30-17:00",
          "Scott_schedule": "Monday: 9:30-15:00, 15:30-16:00, 16:30-17:00; Tuesday: 9:00-9:30, 10:00-11:00, 11:30-12:00, 12:30-13:30, 14:00-15:00, 16:00-16:30; Wednesday: 9:30-12:30, 13:00-13:30, 14:00-14:30, 15:00-15:30, 16:00-16:30; Thursday: 9:00-9:30, 10:00-10:30, 11:00-12:00, 12:30-13:00, 15:00-16:00, 16:30-17:00",
          "Betty_preference": "Not before 15:00 on Monday, Tuesday, Thursday",
          "Scott_preference": "Avoid Wednesday"
        }}
        Available Slots:
        Thursday: 16:00-16:30

        Extracted Data: {data}
        Available Slots:
        """
        return call_llm(prompt, system_instruction)

    available_slots = find_available_slots(extracted_data)
    print(f"Available Slots: {available_slots}")
    return "Here is the proposed time: " + available_slots