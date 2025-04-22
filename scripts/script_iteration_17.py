import os
import re
import math

def main(question):
    """Schedules meetings using a LLM-driven approach that extracts constraints and validates time validity"""
    try:
        # Extract all information regarding the meeting
        extracted_info = extract_meeting_info(question)
        if "Error" in extracted_info:
            return extracted_info
        
        # Propose a valid meeting time
        proposed_time = propose_meeting_time(extracted_info, question)
        if "Error" in proposed_time:
            return proposed_time
        
        return proposed_time
        
    except Exception as e:
        return f"Error processing the request: {str(e)}"
        
def extract_meeting_info(question, max_attempts = 3):
    """Extract the meeting participants, duration, days, and schedules"""
    system_instruction = "You are an expert at extracting meeting details from text. You MUST focus solely on data extraction. Do NOT perform any scheduling logic."
    prompt = f"""
    You are an expert at extracting meeting details from the input text. 
    You must follow the following format: 
    
    Example 1:
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9:00-10:00, Mary is busy from 11:00-12:00.
    Extraction: Participants: John, Mary
              Duration: 30
              Days: Monday
              John: 9:00-10:00
              Mary: 11:00-12:00
    
    Example 2:
    Question: Schedule a meeting for Amy and Kevin for half an hour between the work hours of 9:00 to 17:00 on either Monday, Tuesday or Wednesday. Amy has blocked their calendar on Monday during 11:30 to 12:30, 13:30 to 14:00, 14:30 to 15:00, 16:30 to 17:00, Tuesday during 10:30 to 11:00, 12:30 to 13:00, 13:30 to 14:00, 15:30 to 17:00, Wednesday during 11:30 to 12:00, 15:00 to 15:30, 16:00 to 16:30; Kevin is busy on Monday during 9:00 to 11:00, 11:30 to 17:00, Tuesday during 9:00 to 10:30, 11:00 to 16:30, Wednesday during 9:00 to 9:30, 10:00 to 17:00;
    Extraction: Participants: Amy, Kevin
              Duration: 30
              Days: Monday, Tuesday, Wednesday
              Amy: Monday(11:30-12:30, 13:30-14:00, 14:30-15:00, 16:30-17:00), Tuesday(10:30-11:00, 12:30-13:00, 13:30-14:00, 15:30-17:00), Wednesday(11:30-12:00, 15:00-15:30, 16:00-16:30)
              Kevin: Monday(9:00-11:00, 11:30-17:00), Tuesday(9:00-10:30, 11:00-16:30), Wednesday(9:00-9:30, 10:00-17:00)
              
    Question: {question}
    Extraction: 
    """
    
    extracted_info = call_llm(prompt, system_instruction)
    return extracted_info
    
def propose_meeting_time(extracted_info, question):
    """Proposes the proposed meeting time that satisfies all participants' schedule"""
    system_instruction = "You are an expert at proposing the best meeting schedule time given the extracted info and question."
    prompt = f"""
    You are an expert at proposing a valid meeting time that is within constraints of all given participants:
    You will need to follow the extraction, the participant schedules, and finally derive a valid proposed time
    
    Example 1:
    Extraction: Participants: John, Mary
              Duration: 30
              Days: Monday
              John: 9:00-10:00
              Mary: 11:00-12:00
    Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9:00-10:00, Mary is busy from 11:00-12:00.
    Proposed Time: Here is the proposed time: Monday, 10:00-10:30
    
    Example 2:
    Extraction: Participants: Amy, Kevin
              Duration: 30
              Days: Monday, Tuesday, Wednesday
              Amy: Monday(11:30-12:30, 13:30-14:00, 14:30-15:00, 16:30-17:00), Tuesday(10:30-11:00, 12:30-13:00, 13:30-14:00, 15:30-17:00), Wednesday(11:30-12:00, 15:00-15:30, 16:00-16:30)
              Kevin: Monday(9:00-11:00, 11:30-17:00), Tuesday(9:00-10:30, 11:00-16:30), Wednesday(9:00-9:30, 10:00-17:00)
    Question: Schedule a meeting for Amy and Kevin for half an hour between the work hours of 9:00 to 17:00 on either Monday, Tuesday or Wednesday. Amy has blocked their calendar on Monday during 11:30 to 12:30, 13:30 to 14:00, 14:30 to 15:00, 16:30 to 17:00, Tuesday during 10:30 to 11:00, 12:30 to 13:00, 13:30 to 14:00, 15:30 to 17:00, Wednesday during 11:30 to 12:00, 15:00 to 15:30, 16:00 to 16:30; Kevin is busy on Monday during 9:00 to 11:00, 11:30 to 17:00, Tuesday during 9:00 to 10:30, 11:00 to 16:30, Wednesday during 9:00 to 9:30, 10:00 to 17:00;
    Proposed Time: Here is the proposed time: Wednesday, 9:30-10:00

    Extraction: {extracted_info}
    Question: {question}
    Proposed Time:
    """
    
    proposed_time = call_llm(prompt, system_instruction)
    return proposed_time

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