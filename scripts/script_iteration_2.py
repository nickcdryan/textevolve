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

def extract_info_with_cot(problem):
    """Extract information using Chain of Thought with example."""
    system_instruction = "You are an expert meeting scheduler. Extract relevant information with reasoning."
    prompt = f"""
    Your task is to extract the key information from the scheduling problem, thinking step by step.
    Here's an example:
    Question: Schedule a 30-minute meeting for Alice and Bob on Monday between 9am and 5pm. Alice is busy 10-11am, and Bob is busy 2-3pm.
    Let's think step by step:
    1. Participants: Alice, Bob
    2. Duration: 30 minutes
    3. Day: Monday
    4. Time range: 9am-5pm
    5. Alice's availability: Not available 10-11am
    6. Bob's availability: Not available 2-3pm
    Extracted Information:
    {{
        "participants": ["Alice", "Bob"],
        "duration": "30 minutes",
        "day": "Monday",
        "time_range": "9am-5pm",
        "Alice": ["10-11am"],
        "Bob": ["2-3pm"]
    }}
    Now, extract the information from the following problem:
    {problem}
    """
    return call_llm(prompt, system_instruction)

def find_meeting_time(extracted_info):
    """Find a valid meeting time using extracted info and chain-of-thought."""
    system_instruction = "You are an expert at determining valid meeting times."
    prompt = f"""
    Given the extracted information, find a valid meeting time. Let's think step by step.
    Example:
    Extracted Info:
    {{
        "participants": ["Alice", "Bob"],
        "duration": "30 minutes",
        "day": "Monday",
        "time_range": "9am-5pm",
        "Alice": ["10:00-11:00"],
        "Bob": ["2:00-3:00"]
    }}
    Reasoning:
    1. Available time range: 9am-5pm on Monday
    2. Exclude Alice's busy time: 10:00-11:00
    3. Exclude Bob's busy time: 2:00-3:00
    4. Possible times: 9:00-9:30, 9:30-10:00, 11:00-11:30, ..., 4:30-5:00
    5. Select the earliest available time
    Proposed Time: Monday, 9:00 - 9:30
    Now, with this information:
    {extracted_info}
    Find a valid meeting time.
    """
    return call_llm(prompt, system_instruction)

def verify_meeting_time(problem, proposed_time):
    """Verify if the proposed meeting time is valid."""
    system_instruction = "You are a solution verifier. Determine if a proposed meeting time is valid given the constraints."
    prompt = f"""
    Problem: {problem}
    Proposed Time: {proposed_time}
    
    Let's verify step by step if the proposed time is valid given the problem constraints.
    
    Example:
    Problem: Schedule a 30-minute meeting for Alice and Bob on Monday between 9am and 5pm. Alice is busy 10-11am, and Bob is busy 2-3pm.
    Proposed Time: Monday, 9:00 - 9:30
    
    Verification:
    1. Meeting Length: 30 minutes - OK
    2. Time Range: 9am-5pm - OK
    3. Alice Availability: 9:00-9:30 is not within 10-11am - OK
    4. Bob Availability: 9:00-9:30 is not within 2-3pm - OK
    
    Conclusion: VALID
    
    Now, verify this proposed time:
    Problem: {problem}
    Proposed Time: {proposed_time}
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule a meeting."""
    try:
        extracted_info = extract_info_with_cot(question)
        proposed_time = find_meeting_time(extracted_info)
        verification_result = verify_meeting_time(question, proposed_time)
        if "VALID" in verification_result:
            return proposed_time
        else:
            return "No valid meeting time found."
    except Exception as e:
        return f"Error during scheduling: {str(e)}"