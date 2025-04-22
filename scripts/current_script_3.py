import os
import re
import math

def main(question):
    """Schedules meetings using LLM to extract, analyze, and propose times with validation."""
    try:
        # Step 1: Extract structured info using LLM
        extracted_info = extract_meeting_info(question)

        # Step 2: Analyze the extracted info and participant schedules to identify available time slots using an LLM.
        available_slots = identify_available_time_slots(extracted_info, question)

        # Step 3: Propose a meeting time using LLM and the analyzed data
        proposed_time = propose_meeting_time(available_slots, extracted_info, question)

        # Step 4: Validate the final proposed time for hard constraints
        final_verification = verify_final_solution(proposed_time, extracted_info, question)

        return final_verification

    except Exception as e:
        return f"Error processing the request: {str(e)}"

def extract_meeting_info(question):
    """Extracts structured information from the question using LLM."""
    system_instruction = "You are an expert at extracting meeting details."
    prompt = f"""
    Extract the following meeting details from the question: participants, duration, days, and existing schedules.

    Example 1:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9-10, Mary is busy from 11-12.
    Extracted Info: Participants: John, Mary; Duration: 30; Days: Monday; John's schedule: 9:00-10:00; Mary's schedule: 11:00-12:00

    Example 2:
    Question: Schedule a meeting for Alice, Bob, and Charlie for 1 hour on Tuesday and Wednesday. Alice is busy from 14:00-15:00 on Tuesday, Bob is busy from 10:00-11:00 on Wednesday. Charlie is free.
    Extracted Info: Participants: Alice, Bob, Charlie; Duration: 60; Days: Tuesday, Wednesday; Alice's schedule: 14:00-15:00 (Tuesday); Bob's schedule: 10:00-11:00 (Wednesday); Charlie is free.

    Question: {question}
    Extracted Info:
    """
    return call_llm(prompt, system_instruction)

def identify_available_time_slots(extracted_info, question):
    """Identify available time slots based on extracted information using LLM."""
    system_instruction = "You are an expert at reasoning about schedules to find available time slots and determine hard constraints that prohibit a person's ability to meet."
    prompt = f"""
    Based on these extracted meeting details, analyze each participant's schedule to identify the hard constraints that prevent a meeting.

    Example 1:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9-10, Mary is busy from 11-12.
    Extracted Info: Participants: John, Mary; Duration: 30; Days: Monday; John's schedule: 9:00-10:00; Mary's schedule: 11:00-12:00
    Hard Constraints: John is unavailable from 9:00-10:00; Mary is unavailable from 11:00-12:00

    Example 2:
    Question: Schedule a meeting for Alice, Bob, and Charlie for 1 hour on Tuesday and Wednesday. Alice is busy from 14:00-15:00 on Tuesday, Bob is busy from 10:00-11:00 on Wednesday. Charlie is free.
    Extracted Info: Participants: Alice, Bob, and Charlie; Duration: 60; Days: Tuesday and Wednesday; Alice's schedule: 14:00-15:00 (Tuesday); Bob's schedule: 10:00-11:00 (Wednesday); Charlie is free.
    Hard Constraints: Alice is unavailable from 14:00-15:00 on Tuesday; Bob is unavailable from 10:00-11:00 on Wednesday.

    Question: {question}
    Extracted Info: {extracted_info}
    Hard Constraints:
    """
    return call_llm(prompt, system_instruction)

def propose_meeting_time(available_slots, extracted_info, question):
    """Propose a suitable meeting time based on available slots and participant constraints."""
    system_instruction = "You are skilled at proposing meeting times considering participant constraints, and you are also able to give that time in the format 'Here is the proposed time: [day], [start_time]-[end_time]'."
    prompt = f"""
    Considering these constraints and meeting details, propose the best meeting time. Respond in the format 'Here is the proposed time: [day], [start_time]-[end_time]'

    Example 1:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9-10, Mary is busy from 11-12.
    Hard Constraints: John is unavailable from 9:00-10:00; Mary is unavailable from 11:00-12:00
    Extracted Info: Participants: John, Mary; Duration: 30; Days: Monday
    Proposed Time: Here is the proposed time: Monday, 10:00-10:30

    Example 2:
    Question: Schedule a meeting for Alice, Bob, and Charlie for 1 hour on Tuesday and Wednesday. Alice is busy from 14:00-15:00 on Tuesday, Bob is busy from 10:00-11:00 on Wednesday. Charlie is free.
    Hard Constraints: Alice is unavailable from 14:00-15:00 on Tuesday; Bob is unavailable from 10:00-11:00 on Wednesday
    Extracted Info: Participants: Alice, Bob, Charlie; Duration: 60; Days: Tuesday, Wednesday
    Proposed Time: Here is the proposed time: Tuesday, 10:00-11:00

    Question: {question}
    Hard Constraints: {available_slots}
    Extracted Info: {extracted_info}
    Proposed Time:
    """
    return call_llm(prompt, system_instruction)

def verify_final_solution(proposed_time, extracted_info, question):
    """Verify if the proposed time works with everyone's schedule and constraints."""
    system_instruction = "You are an expert verifier who makes sure the proposed solution works with all the scheduling constraints, and can determine why a proposed time is correct."
    prompt = f"""
    Verify the proposed meeting time against all the given constraints and participant schedules and provide a reason why the proposed time does work.

    Example 1:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9-10, Mary is busy from 11-12.
    Proposed Time: Here is the proposed time: Monday, 10:30-11:00
    Extracted Info: Participants: John, Mary; Duration: 30; Days: Monday; John's schedule: 9:00-10:00; Mary's schedule: 11:00-12:00
    Verification: The proposed time works for both John and Mary because it is outside of both of their busy times.

    Example 2:
    Question: Schedule a meeting for Alice, Bob, and Charlie for 1 hour on Tuesday and Wednesday. Alice is busy from 14:00-15:00 on Tuesday, Bob is busy from 10:00-11:00 on Wednesday. Charlie is free.
    Proposed Time: Here is the proposed time: Tuesday, 10:00-11:00
    Extracted Info: Participants: Alice, Bob, Charlie; Duration: 60; Days: Tuesday, Wednesday; Alice's schedule: 14:00-15:00 (Tuesday); Bob's schedule: 10:00-11:00 (Wednesday); Charlie is free.
    Verification: The proposed time works for Alice, Bob, and Charlie because it is outside of Alice's Tuesday busy time, Bob is free, and Charlie is free.

    Question: {question}
    Proposed Time: {proposed_time}
    Extracted Info: {extracted_info}
    Verification:
    """
    return call_llm(prompt, system_instruction)

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