import os
import re
import math

def main(question):
    """
    Schedules meetings using a structured approach with two specialized agents and multi-stage verification.
    """
    try:
        # Step 1: Extract meeting information using the Extraction Agent with validation
        extracted_info = extract_meeting_info(question)
        if "Error" in extracted_info:
            return extracted_info

        # Step 2: Schedule the meeting using the Scheduling Agent with validation
        scheduled_meeting = schedule_meeting(extracted_info, question)
        if "Error" in scheduled_meeting:
            return scheduled_meeting

        return scheduled_meeting

    except Exception as e:
        return f"Error processing the request: {str(e)}"

def extract_meeting_info(question, max_attempts=3):
    """Extracts meeting details (participants, duration, days, schedules) using a specialized extraction agent with multi-example prompting and verification."""
    system_instruction = "You are an expert at extracting meeting details from text. Your only job is to extract data, not to determine if the time works."

    for attempt in range(max_attempts):
        prompt = f"""
        You are an expert at extracting meeting details from text. Your goal is to pull out the important information. Your only job is to extract data, not to determine if the time works. Extract:
        - participants (list of names)
        - duration (integer, minutes)
        - days (list of strings, e.g., "Monday", "Tuesday")
        - existing schedules (dictionary, participant name -> list of time ranges "HH:MM-HH:MM")

        Example 1:
        Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9:00-10:00, Mary is busy from 11:00-12:00.
        Extraction: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": ["9:00-10:00"], "Mary": ["11:00-12:00"]}}}}

        Example 2:
        Question: Schedule a meeting for Alice, Bob, and Charlie for 1 hour on Tuesday and Wednesday. Alice is busy from 14:00-15:00 on Tuesday, Bob is busy from 10:00-11:00 on Wednesday. Charlie is free.
        Extraction: {{"participants": ["Alice", "Bob", "Charlie"], "duration": 60, "days": ["Tuesday", "Wednesday"], "schedules": {{"Alice": ["14:00-15:00 (Tuesday)"], "Bob": ["10:00-11:00 (Wednesday)"], "Charlie": []}}}}

        Example 3:
        Question: You need to schedule a meeting for Jonathan, Janice, Walter, Mary, Roger, Tyler and Arthur for half an hour between the work hours of 9:00 to 17:00 on Monday. Jonathan has meetings on Monday during 9:30 to 10:00, 12:30 to 13:30, 14:30 to 15:00; Janice has blocked their calendar on Monday during 9:00 to 9:30, 11:30 to 12:00, 12:30 to 13:30, 14:30 to 15:00, 16:00 to 16:30; Walter has blocked their calendar on Monday during 9:30 to 10:00, 11:30 to 12:00; Mary is busy on Monday during 12:00 to 12:30, 13:30 to 14:00; Roger has blocked their calendar on Monday during 9:30 to 10:30, 11:00 to 12:30, 13:00 to 13:30, 14:00 to 15:30, 16:00 to 16:30; Tyler has blocked their calendar on Monday during 9:30 to 11:00, 11:30 to 12:30, 13:30 to 14:00, 15:00 to 16:00; Arthur is busy on Monday during 10:00 to 11:30, 12:30 to 13:00, 13:30 to 14:00, 14:30 to 16:00;
        Extraction: {{"participants": ["Jonathan", "Janice", "Walter", "Mary", "Roger", "Tyler", "Arthur"], "duration": 30, "days": ["Monday"], "schedules": {{"Jonathan": ["9:30-10:00", "12:30-13:30", "14:30-15:00"], "Janice": ["9:00-9:30", "11:30-12:00", "12:30-13:30", "14:30-15:00", "16:00-16:30"], "Walter": ["9:30-10:00", "11:30-12:00"], "Mary": ["12:00-12:30", "13:30-14:00"], "Roger": ["9:30-10:30", "11:00-12:30", "13:00-13:30", "14:00-15:30", "16:00-16:30"], "Tyler": ["9:30-11:00", "11:30-12:30", "13:30-14:00", "15:00-16:00"], "Arthur": ["10:00-11:30", "12:30-13:00", "13:30-14:00", "14:30-16:00"]}}}}

        Question: {question}
        Extraction:
        """
        extracted_info = call_llm(prompt, system_instruction)

        # Validation step
        validation_prompt = f"""
        You are an expert at verifying extracted information. Given the question and the extraction, verify:
        1. Are all participants identified?
        2. Is the duration correct?
        3. Are all days mentioned included?
        4. Are the schedules correctly associated with each participant and day?

        If EVERYTHING is correct, respond EXACTLY with "VALID".
        Otherwise, explain the errors.

        Question: {question}
        Extracted Info: {extracted_info}
        Verification:
        """
        validation_result = call_llm(validation_prompt, system_instruction)
        if "VALID" in validation_result:
            return extracted_info
        else:
            print(f"Extraction validation failed (attempt {attempt+1}): {validation_result}")
    return f"Error: Extraction failed after multiple attempts: {validation_result}"

def schedule_meeting(extracted_info, question, max_attempts=3):
    """Schedules a meeting given extracted information with validation and retry."""
    system_instruction = "You are an expert meeting scheduler. You are given all the information and must generate a final time that works."

    for attempt in range(max_attempts):
        prompt = f"""
        You are an expert at scheduling meetings. Given the question and the extracted meeting details, your goal is to return a final proposed time that satisfies all constraints.
        You are given the following information:
        - Participants: list of names
        - Duration: integer, minutes
        - Days: list of strings, e.g., "Monday", "Tuesday"
        - Existing schedules: dictionary, participant name -> list of time ranges "HH:MM-HH:MM"

        Example 1:
        Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9:00-10:00, Mary is busy from 11:00-12:00.
        Extraction: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": ["9:00-10:00"], "Mary": ["11:00-12:00"]}}}}
        Reasoning: John is available from 10:00-17:00. Mary is available from 9:00-11:00 and 12:00-17:00. The best available time that works for both is 10:00-10:30.
        Proposed Time: Here is the proposed time: Monday, 10:00-10:30

        Example 2:
        Question: Schedule a meeting for Alice, Bob, and Charlie for 1 hour on Tuesday and Wednesday. Alice is busy from 14:00-15:00 on Tuesday, Bob is busy from 10:00-11:00 on Wednesday. Charlie is free.
        Extraction: {{"participants": ["Alice", "Bob", "Charlie"], "duration": 60, "days": ["Tuesday", "Wednesday"], "schedules": {{"Alice": ["14:00-15:00 (Tuesday)"], "Bob": ["10:00-11:00 (Wednesday)"], "Charlie": []}}}}
        Reasoning: On Tuesday, Alice is busy from 9:00-14:00 and 15:00-17:00, Bob and Charlie are free. A time that works for all is 9:00-10:00.
        Proposed Time: Here is the proposed time: Tuesday, 9:00-10:00

        Example 3:
        Question: You need to schedule a meeting for Stephanie, Mark and Scott for one hour between the work hours of 9:00 to 17:00 on Monday. Stephanie has blocked their calendar on Monday during 9:00 to 9:30, 13:30 to 14:00; Mark's calendar is wide open the entire day. Scott is busy on Monday during 9:00 to 10:00, 11:00 to 12:30, 14:30 to 15:00, 16:00 to 17:00; Mark would like to avoid more meetings on Monday before 15:00.
        Extraction: {{"participants": ["Stephanie", "Mark", "Scott"], "duration": 60, "days": ["Monday"], "schedules": {{"Stephanie": ["9:00-9:30", "13:30-14:00"], "Mark": [], "Scott": ["9:00-10:00", "11:00-12:30", "14:30-15:00", "16:00-17:00"]}}}}
        Reasoning: Stephanie is available from 9:30-13:30 and 14:00-17:00, Mark is free, and Scott is available from 10:00-11:00, 12:30-14:30 and 15:00-16:00. Given Mark's preferences, the ideal time is 15:00-16:00.
        Proposed Time: Here is the proposed time: Monday, 15:00-16:00

        Considering the above, determine an appropriate meeting time given this extracted information and the question.
        Extracted Info: {extracted_info}
        Question: {question}

        Respond in the format 'Here is the proposed time: [day], [start_time]-[end_time]'
        Proposed Time:
        """
        proposed_time = call_llm(prompt, system_instruction)

        # Verification step: check that it's in the correct format
        if not re.match(r"Here is the proposed time: \w+, \d{1,2}:\d{2}-\d{1,2}:\d{2}", proposed_time):
            print(f"Scheduling failed (attempt {attempt+1}): Incorrect format")
            continue  # Retry if format is incorrect

        #Improved Error Handling (Specific Exception Handling):
        try:
             #Extract day, start_time, end_time safely
            match = re.search(r"Here is the proposed time: (\w+), (\d{1,2}:\d{2})-(\d{1,2}:\d{2})", proposed_time)
            if match:
                day, start_time, end_time = match.groups()

                #Time format verification (e.g., 9:00 to be padded as 09:00, and 17:00 is valid)
                if not (re.match(r"^\d{1,2}:\d{2}$", start_time) and re.match(r"^\d{1,2}:\d{2}$", end_time)):
                    print(f"Scheduling failed (attempt {attempt+1}): Incorrect time format")
                    continue

                return proposed_time #Valid time found, return it
            else:
                print(f"Scheduling failed (attempt {attempt+1}): No match found. Returning a failed message.")
                return "Error: Could not extract schedule time."

        except ValueError as ve:
            print(f"Value Error: Scheduling failed (attempt {attempt+1}): {str(ve)}")
            continue #Retry scheduling if there is a value issue.

        except Exception as e: #Catch all for other potential problems.
            print(f"General Error: Scheduling failed (attempt {attempt+1}): {str(e)}")
            continue

    return "Error: Scheduling failed after multiple attempts."

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