import os
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

def extract_info_structured(question):
    """Extract meeting info, participants, and schedules using structured prompting and examples."""
    system_instruction = "You are an expert at extracting structured information for meeting scheduling."
    prompt = f"""
    Extract the following information from the text, structured as JSON:
    - participants: A list of all participants in the meeting.
    - duration: The duration of the meeting (e.g., "half an hour", "one hour").
    - days: A list of possible days for the meeting.
    - schedules: A dictionary where keys are participant names, and values are lists of busy time intervals (e.g., "9:00-10:00").

    Example:
    Text: You need to schedule a meeting for Daniel, Kathleen, and Carolyn for half an hour on Monday. Daniel is busy 9:00-9:30, Kathleen is busy 10:00-11:00, and Carolyn is busy 14:00-15:00.
    Output:
    {{
      "participants": ["Daniel", "Kathleen", "Carolyn"],
      "duration": "half an hour",
      "days": ["Monday"],
      "schedules": {{
        "Daniel": ["9:00-9:30"],
        "Kathleen": ["10:00-11:00"],
        "Carolyn": ["14:00-15:00"]
      }}
    }}

    Now, extract the structured information from the following text:
    {question}
    """
    try:
        response = call_llm(prompt, system_instruction)
        return json.loads(response)
    except Exception as e:
        print(f"Error extracting structured info: {e}")
        return None

def verify_extracted_info(extracted_info, question):
    """Verify the extracted information against the original question using LLM."""
    system_instruction = "You are a meticulous verifier, checking information extraction for accuracy."
    prompt = f"""
    You are provided with structured information extracted from a text and the original text.
    Verify that the extracted information is complete, accurate, and consistent with the original text.
    If there are any errors or omissions, explain them in detail. If the extraction is perfect, respond with "VALID".

    Example:
    Text: Schedule a meeting for John and Jane on Tuesday. John is busy 9:00-10:00.
    Extracted Info:
    {{
      "participants": ["John"],
      "duration": null,
      "days": ["Tuesday"],
      "schedules": {{
        "John": ["9:00-10:00"]
      }}
    }}
    Verification Result:
    "INVALID: Missing participant 'Jane'. Duration is not specified."

    Text: Schedule a meeting for Alice and Bob for one hour on Wednesday. Alice is busy 11:00-12:00, Bob is busy 14:00-15:00.
    Extracted Info:
    {{
      "participants": ["Alice", "Bob"],
      "duration": "one hour",
      "days": ["Wednesday"],
      "schedules": {{
        "Alice": ["11:00-12:00"],
        "Bob": ["14:00-15:00"]
      }}
    }}
    Verification Result: VALID

    Now, verify the extracted info against the following text:
    Text: {question}
    Extracted Info: {json.dumps(extracted_info)}
    """
    try:
        response = call_llm(prompt, system_instruction)
        return response
    except Exception as e:
        print(f"Error verifying extracted info: {e}")
        return None

def find_available_time(extracted_info):
    """Find an available meeting time using LLM, given the extracted and verified info."""
    system_instruction = "You are an expert at determining available meeting times based on participant schedules."
    prompt = f"""
    Given the extracted meeting information, determine an available time slot that works for all participants, considering their schedules and any other constraints.
    Respond in the format: "Here is the proposed time: [Day], [Start Time] - [End Time]". If no time is available, respond with "No available time found."

    Example:
    Extracted Info:
    {{
      "participants": ["Daniel", "Kathleen"],
      "duration": "half an hour",
      "days": ["Monday"],
      "schedules": {{
        "Daniel": ["9:00-9:30"],
        "Kathleen": ["10:00-11:00"]
      }}
    }}
    Available Time: Here is the proposed time: Monday, 9:30 - 10:00

    Now, find an available time for the following extracted info:
    {json.dumps(extracted_info)}
    """
    try:
        response = call_llm(prompt, system_instruction)
        return response
    except Exception as e:
        print(f"Error finding available time: {e}")
        return None

def main(question):
    """Main function to schedule a meeting."""
    try:
        # 1. Extract structured information
        extracted_info = extract_info_structured(question)
        if not extracted_info:
            return "Error: Could not extract information."

        # 2. Verify extracted information
        verification_result = verify_extracted_info(extracted_info, question)
        if "VALID" not in verification_result:
            return f"Error: Information extraction verification failed: {verification_result}"

        # 3. Find available time
        available_time = find_available_time(extracted_info)
        if not available_time:
            return "Error: Could not find available time."

        return available_time

    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred while scheduling the meeting."