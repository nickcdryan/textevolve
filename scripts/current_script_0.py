import os
import json
import re
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

def extract_meeting_info(question, max_attempts=3):
    """Extract key meeting information from the input question with verification loop."""
    system_instruction = "You are an expert at extracting meeting details."

    for attempt in range(max_attempts):
        prompt = f"""
        Extract the following information from the text:
        - Participants: List of people involved in the meeting.
        - Duration: Length of the meeting (e.g., "half an hour").
        - Time Constraints: Working hours (e.g., "9:00 to 17:00"), specific days.
        - Existing Schedules: Schedules of participants (e.g., "Joyce has meetings on Monday during 11:00 to 11:30").
        - Preferences: Preferences of participants (e.g., "Natalie would rather not meet on Monday after 10:30").

        Example 1:
        Text: You need to schedule a meeting for Joyce, Christine and Alexander for half an hour between the work hours of 9:00 to 17:00 on Monday. Joyce has meetings on Monday during 11:00 to 11:30; Christine has no meetings the whole day. Alexander has meetings on Monday during 9:00 to 11:00.
        Extracted Information:
        {{
          "Participants": ["Joyce", "Christine", "Alexander"],
          "Duration": "half an hour",
          "Time Constraints": "between 9:00 to 17:00 on Monday",
          "Existing Schedules": "Joyce: Monday during 11:00 to 11:30; Christine: no meetings; Alexander: Monday during 9:00 to 11:00",
          "Preferences": "None"
        }}

        Example 2:
        Text: You need to schedule a meeting for Betty and Scott for half an hour between the work hours of 9:00 to 17:00 on either Monday, Tuesday, Wednesday or Thursday. Betty is busy on Monday during 10:00 to 10:30, Tuesday during 9:00 to 9:30; Scott is busy on Monday during 9:30 to 15:00, Tuesday during 9:00 to 9:30. Betty can not meet on Monday. Tuesday. Thursday before 15:00.
        Extracted Information:
        {{
          "Participants": ["Betty", "Scott"],
          "Duration": "half an hour",
          "Time Constraints": "between 9:00 to 17:00 on Monday, Tuesday, Wednesday or Thursday",
          "Existing Schedules": "Betty: Monday during 10:00 to 10:30, Tuesday during 9:00 to 9:30; Scott: Monday during 9:30 to 15:00, Tuesday during 9:00 to 9:30",
          "Preferences": "Betty can not meet on Monday, Tuesday, Thursday before 15:00"
        }}

        Text: {question}
        Extracted Information:
        """

        extracted_info = call_llm(prompt, system_instruction)

        #Verification step
        verification_prompt = f"""
        You have extracted the following information from the problem text:
        {extracted_info}
        Check if the information extracted is accurate and complete for scheduling a meeting.
        Respond 'VALID' if all key pieces of scheduling information is included.
        Respond 'INVALID' with the missing information mentioned if it is not valid.
        """
        verification_result = call_llm(verification_prompt, system_instruction)

        if "VALID" in verification_result:
            try:
                return json.loads(extracted_info)
            except:
                #Attempt to fix JSON formatting if parsing fails.
                json_fix_prompt = f"""The json you returned is not properly formatted. Can you fix it?
                {extracted_info}
                """
                extracted_info = call_llm(json_fix_prompt, system_instruction)
                return json.loads(extracted_info)
        else:
            print(f"Attempt {attempt + 1} failed. Validation feedback: {verification_result}")
            if attempt == max_attempts - 1:
                return {"Error": verification_result}

def schedule_meeting(meeting_info):
    """Schedule the meeting based on the extracted information."""
    system_instruction = "You are an expert at scheduling meetings given constraints."

    prompt = f"""
    Given the following meeting information, find a suitable time slot:
    {json.dumps(meeting_info, indent=2)}

    Consider all constraints and preferences to find the best possible time.
    Provide the result as a complete sentence starting with "Here is the proposed time:".

    Example:
    Meeting Information:
    {{
      "Participants": ["Joyce", "Christine", "Alexander"],
      "Duration": "half an hour",
      "Time Constraints": "between 9:00 to 17:00 on Monday",
      "Existing Schedules": "Joyce: Monday during 11:00 to 11:30; Christine: no meetings; Alexander: Monday during 9:00 to 11:00",
      "Preferences": "None"
    }}
    Proposed Time: Here is the proposed time: Monday, 12:30 - 13:00

    Example:
    Meeting Information:
    {{
      "Participants": ["Betty", "Scott"],
      "Duration": "half an hour",
      "Time Constraints": "between 9:00 to 17:00 on Monday, Tuesday, Wednesday or Thursday",
      "Existing Schedules": "Betty: Monday during 10:00 to 10:30, Tuesday during 9:00 to 9:30; Scott: Monday during 9:30 to 15:00, Tuesday during 9:00 to 9:30",
      "Preferences": "Betty can not meet on Monday, Tuesday, Thursday before 15:00"
    }}
    Proposed Time: Here is the proposed time: Thursday, 16:00 - 16:30
    """

    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to process the question and return the answer."""
    try:
        meeting_info = extract_meeting_info(question)
        if "Error" in meeting_info:
            return f"Error: {meeting_info['Error']}"
        answer = schedule_meeting(meeting_info)
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

#Example Usage
#question = "You need to schedule a meeting for Joyce, Christine and Alexander for half an hour between the work hours of 9:00 to 17:00 on Monday. Joyce has meetings on Monday during 11:00 to 11:30, 13:30 to 14:00, 14:30 to 16:30; Christine has no meetings the whole day. Alexander has meetings on Monday during 9:00 to 11:00, 12:00 to 12:30, 13:30 to 15:00, 15:30 to 16:00, 16:30 to 17:00; Christine can not meet on Monday before 12:00. Find a time that works for everyone's schedule and constraints."
#answer = main(question)
#print(answer)